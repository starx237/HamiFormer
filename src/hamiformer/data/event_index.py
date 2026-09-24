from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from torch.utils.data import Dataset
from .dataset import PhaseWindowDataset

@dataclass(frozen=True)
class EventIndex:
    substeps_by_sample: dict[str, tuple[int, ...]]

    @classmethod
    def from_jsonl(cls, path: str | Path) -> 'EventIndex':
        event_path = Path(path)
        if not event_path.is_file():
            raise FileNotFoundError(f'event sidecar 不存在: {event_path}')
        result: dict[str, tuple[int, ...]] = {}
        with event_path.open('r', encoding='utf-8') as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise ValueError(f'event sidecar 第 {line_number} 行必须是 mapping')
                sample_id = raw.get('sample_id')
                events = raw.get('events')
                if not isinstance(sample_id, str) or not sample_id:
                    raise ValueError(f'event sidecar 第 {line_number} 行 sample_id 非法')
                if sample_id in result:
                    raise ValueError(f'event sidecar sample_id 重复: {sample_id}')
                if not isinstance(events, list):
                    raise ValueError(f'event sidecar 第 {line_number} 行 events 必须是 list')
                substeps: list[int] = []
                for event_index, event in enumerate(events):
                    if not isinstance(event, dict):
                        raise ValueError(f'event sidecar 第 {line_number} 行第 {event_index} 个事件非法')
                    substep = event.get('substep')
                    if type(substep) is not int or substep < 0:
                        raise ValueError(f'event sidecar 第 {line_number} 行含非法 substep: {substep!r}')
                    substeps.append(substep)
                result[sample_id] = tuple(sorted(substeps))
        if not result:
            raise ValueError(f'event sidecar 为空: {event_path}')
        return cls(substeps_by_sample=result)

    def validate_sample_ids(self, sample_ids: set[str]) -> None:
        indexed = set(self.substeps_by_sample)
        missing = sample_ids - indexed
        extra = indexed - sample_ids
        if missing or extra:
            raise ValueError(f'manifest/event sidecar sample_id 不一致；缺失={sorted(missing)[:3]}，额外={sorted(extra)[:3]}')

    def has_event(self, sample_id: str, *, start_substep: int=0, end_substep: int | None=None) -> bool:
        if start_substep < 0:
            raise ValueError('start_substep 不能为负')
        if end_substep is not None and end_substep <= start_substep:
            raise ValueError('end_substep 必须大于 start_substep')
        if sample_id not in self.substeps_by_sample:
            raise KeyError(f'event sidecar 不含 sample_id: {sample_id}')
        return any((substep >= start_substep and (end_substep is None or substep < end_substep) for substep in self.substeps_by_sample[sample_id]))

class EventFilteredPhaseDataset(Dataset[dict[str, Any]]):

    def __init__(self, dataset: PhaseWindowDataset, event_index: EventIndex, *, start_substep: int, keep_eventful: bool) -> None:
        sample_ids = {record['sample_id'] for record in dataset.records}
        event_index.validate_sample_ids(sample_ids)
        self.dataset = dataset
        self.event_index = event_index
        self.start_substep = int(start_substep)
        self.keep_eventful = bool(keep_eventful)
        self.indices = [index for index, record in enumerate(dataset.records) if event_index.has_event(record['sample_id'], start_substep=self.start_substep) is self.keep_eventful]
        if not self.indices:
            label = '有事件' if keep_eventful else '无事件'
            raise ValueError(f'筛选后没有{label}样本；start_substep={start_substep}')

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.dataset[self.indices[index]]
