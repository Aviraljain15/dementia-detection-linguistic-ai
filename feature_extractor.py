import spacy
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict

# Load once (good for performance)
nlp = spacy.load('en_core_web_sm')
class LinguisticFeatureExtractor:
    PIU_KEYWORDS = {
        'boy', 'girl', 'woman', 'mother', 'cookie', 'stool', 'jar',
        'kitchen', 'sink', 'dish', 'water', 'overflow', 'steal',
        'fall', 'window', 'outside', 'curtain', 'cabinet', 'plate',
        'reach', 'tippy', 'apron', 'summer', 'hand', 'cup'
    }
    def extract(self, sample) -> Dict[str, float]:
        text = sample.clean_text.strip()
        if not text:
            return self._empty_features()
        doc = nlp(text)
        tokens = [t for t in doc if not t.is_punct]
        words = [t.text.lower() for t in tokens]
        N = len(words)
        feats = {}
        # ── Lexical Features ───────────────────── #
        V = len(set(words))
        feats['ttr'] = V / N if N > 0 else 0
        feats['mattr'] = self._mattr(words, window=50)
        feats['brunets_w'] = N ** (V ** -0.165) if V > 0 else 0
        freq = Counter(words)
        hapax = sum(1 for w in freq if freq[w] == 1)
        if V > hapax and V > 0:
            feats['honores_r'] = (100 * np.log(N + 1e-8)) / (1 - hapax / V)
        else:
            feats['honores_r'] = 0
        # ── POS Features ───────────────────── #
        pos_counts = Counter(t.pos_ for t in doc)

        n_nouns = pos_counts.get('NOUN', 0) + pos_counts.get('PROPN', 0)
        n_verbs = pos_counts.get('VERB', 0) + pos_counts.get('AUX', 0)

        feats['noun_verb_ratio'] = n_nouns / (n_verbs + 1e-8)
        feats['content_word_ratio'] = (
            n_nouns + n_verbs + pos_counts.get('ADJ', 0)
        ) / (N + 1e-8)

        feats['pronoun_ratio'] = pos_counts.get('PRON', 0) / (n_nouns + 1e-8)

        # ── Syntactic Features ───────────────── #
        sentences = list(doc.sents)

        if sentences:
            mlus = [len([t for t in s if not t.is_punct]) for s in sentences]
            feats['mlu'] = np.mean(mlus)

            depths = [self._tree_depth(s.root) for s in sentences]
            feats['parse_depth'] = np.mean(depths)
        else:
            feats['mlu'] = 0
            feats['parse_depth'] = 0

        subord_deps = {'advcl', 'csubj', 'ccomp', 'relcl', 'acl'}
        subord = sum(1 for t in doc if t.dep_ in subord_deps)
        feats['subord_ratio'] = subord / (len(sentences) + 1e-8)

        # ── Disfluency Features ─────────────── #
        disf = sample.disfluency_counts
        total = max(N, 1)

        feats['filler_rate'] = disf['fillers'] / total
        feats['repetition_rate'] = disf['repetitions'] / max(len(sentences), 1)
        feats['pause_rate'] = (
            disf['short_pauses'] + disf['long_pauses']
        ) / total
        feats['incomplete_rate'] = disf['incomplete'] / max(len(sentences), 1)

        # ── PIU Score (only for Cookie Theft) ─ #
        if sample.task == 'cookie_theft':
            found = sum(1 for kw in self.PIU_KEYWORDS if kw in words)
            feats['piu_score'] = found / len(self.PIU_KEYWORDS)
        else:
            feats['piu_score'] = 0

        # ── Discourse Coherence ─────────────── #
        feats['coherence'] = self._coherence(sentences)

        return feats

    # ================= HELPERS ================= #

    def _empty_features(self):
        return {
            'ttr': 0, 'mattr': 0, 'brunets_w': 0, 'honores_r': 0,
            'noun_verb_ratio': 0, 'content_word_ratio': 0, 'pronoun_ratio': 0,
            'mlu': 0, 'parse_depth': 0, 'subord_ratio': 0,
            'filler_rate': 0, 'repetition_rate': 0,
            'pause_rate': 0, 'incomplete_rate': 0,
            'piu_score': 0, 'coherence': 0
        }

    def _mattr(self, words, window=50):
        if len(words) < window:
            return len(set(words)) / (len(words) + 1e-8)

        ttrs = [
            len(set(words[i:i+window])) / window
            for i in range(len(words) - window + 1)
        ]
        return np.mean(ttrs)

    def _tree_depth(self, token, depth=0):
        children = list(token.children)
        if not children:
            return depth
        return max(self._tree_depth(c, depth + 1) for c in children)

    def _coherence(self, sentences):
        if len(sentences) < 2:
            return 1.0

        vecs = [s.vector.reshape(1, -1) for s in sentences if s.has_vector]

        if len(vecs) < 2:
            return 1.0

        sims = [
            cosine_similarity(vecs[i], vecs[i + 1])[0][0]
            for i in range(len(vecs) - 1)
        ]

        return np.mean(sims)