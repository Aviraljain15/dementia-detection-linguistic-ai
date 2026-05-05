import re
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class TranscriptSample:
    participant_id: str
    task: str
    label: int                    # 0 = control, 1 = dementia
    raw_par_turns: List[str] = field(default_factory=list)
    clean_text: str = ""
    disfluency_counts: Dict[str, int] = field(default_factory=dict)


class ChatParser:
    """Parse CHAT-format transcripts from DementiaBank Pitt Corpus."""

    # ---------------- REGEX PATTERNS ---------------- #
    FILLER_PATTERN = re.compile(r'&(?:uh|um|ah|er|hm)\b', re.IGNORECASE)
    REPETITION_PATTERN = re.compile(r'\[//?\]')
    RETRACING_PATTERN = re.compile(r'<[^>]*>\s*\[/\]')
    PAUSE_SHORT = re.compile(r'\(\.\)')
    PAUSE_LONG = re.compile(r'\(\.{2,}\)')
    INCOMPLETE = re.compile(r'\+\.\.\.')
    UNCERTAIN = re.compile(r'\[\?\]')

    # ---------------- MAIN PARSER ---------------- #
    def parse_file(self, path: str, label: int) -> TranscriptSample:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()

        task = self._extract_task(path)
        pid = self._extract_pid(lines)

        # Extract only participant speech
        par_turns = [l.strip() for l in lines if l.strip().startswith('*PAR:')]
        raw_text = ' '.join(par_turns)

        sample = TranscriptSample(
            participant_id=pid,
            task=task,
            label=label
        )

        sample.raw_par_turns = par_turns
        sample.disfluency_counts = self._count_disfluencies(raw_text)
        sample.clean_text = self._clean(raw_text)

        return sample

    # ---------------- DISFLUENCY COUNTS ---------------- #
    def _count_disfluencies(self, text: str) -> Dict[str, int]:
        return {
            'fillers': len(self.FILLER_PATTERN.findall(text)),
            'repetitions': len(self.REPETITION_PATTERN.findall(text)),
            'retracings': len(self.RETRACING_PATTERN.findall(text)),
            'short_pauses': len(self.PAUSE_SHORT.findall(text)),
            'long_pauses': len(self.PAUSE_LONG.findall(text)),
            'incomplete': len(self.INCOMPLETE.findall(text)),
            'uncertain': len(self.UNCERTAIN.findall(text)),
        }

    # ---------------- TEXT CLEANING ---------------- #
    def _clean(self, text: str) -> str:
        # Remove speaker tag
        text = text.replace('*PAR:', '')

        # Remove retracing segments
        text = self.RETRACING_PATTERN.sub('', text)

        # Remove fillers
        text = self.FILLER_PATTERN.sub('', text)

        # Remove CHAT annotations like [/] [//] [?]
        text = re.sub(r'\[.*?\]', ' ', text)

        # Remove pauses like (.) (...)
        text = re.sub(r'\(\.*\)', ' ', text)

        # Remove incomplete markers
        text = re.sub(r'\+\.\.\.', ' ', text)

        # Keep only alphabets
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)

        # Normalize spaces + lowercase
        text = re.sub(r'\s+', ' ', text).strip().lower()

        return text

    # ---------------- TASK EXTRACTION ---------------- #
    def _extract_task(self, path: str) -> str:
        parts = path.replace('\\', '/').split('/')
        task_map = {
            'cookie': 'cookie_theft',
            'fluency': 'verbal_fluency',
            'sentence': 'sentence_construction',
            'recall': 'story_recall',
        }
        for part in parts:
            if part.lower() in task_map:
                return task_map[part.lower()]
        return 'unknown'

    # ---------------- PARTICIPANT ID ---------------- #
    def _extract_pid(self, lines: List[str]) -> str:
        """
        Extract participant ID from @ID line
        Example:
        @ID: eng|Pitt|PAR||female|76;|Alzheimer|MCI||
        """
        for line in lines:
            if line.startswith("@ID:"):
                parts = line.strip().split("|")
                if len(parts) > 2:
                    return parts[1] + "_" + parts[4] + "_" + parts[5]
        return "unknown"