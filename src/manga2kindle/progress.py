"""Progress tracking abstractions and Rich Live terminal reporting for manga2kindle."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import TracebackType
from typing import Protocol, runtime_checkable

from rich.console import Console, Group
from rich.live import Live
from rich.markup import escape as rich_escape
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


class ConversionStage(str, Enum):
    """Distinct execution stages within a single-book conversion or preview run."""

    EXTRACTING = "extracting"
    PREPROCESSING = "preprocessing"
    DETECTING = "detecting"
    PACKAGING = "packaging"
    COMPILING = "compiling"
    RENDERING = "rendering"


STAGE_LABELS: dict[ConversionStage, str] = {
    ConversionStage.EXTRACTING: "Extracting archive",
    ConversionStage.PREPROCESSING: "Preprocessing pages",
    ConversionStage.DETECTING: "Detecting panels",
    ConversionStage.PACKAGING: "Packaging KPF",
    ConversionStage.COMPILING: "Compiling KFX",
    ConversionStage.RENDERING: "Rendering previews",
}


@dataclass(frozen=True)
class ProgressEvent:
    """Immutable stage and optional page-count update for a single book."""

    book_id: str
    book_name: str
    stage: ConversionStage
    completed: int | None = None
    total: int | None = None

    def format_detail(self) -> str:
        """Format stage label and page progress into a human-readable status string."""
        label = STAGE_LABELS[self.stage]
        if self.completed is not None and self.total is not None and self.total > 0:
            return f"{label} ({self.completed}/{self.total} pages)"
        if self.total is not None and self.total > 0:
            return f"{label} ({self.total} pages)"
        return label


@runtime_checkable
class ProgressReporter(Protocol):
    """Interface for reporting batch, per-book, and per-stage conversion progress."""

    def on_batch_start(self, total: int, *, label: str = "Converting") -> None:
        """Signal the start of a batch job covering ``total`` archives."""
        ...

    def on_book_start(self, book_id: str, book_name: str) -> None:
        """Signal that processing has begun for an archive or folder."""
        ...

    def on_book_stage(self, event: ProgressEvent) -> None:
        """Report a stage transition or page-level progress update for a book."""
        ...

    def on_book_complete(self, book_id: str, book_name: str, output_path: Path) -> None:
        """Signal successful completion for a book."""
        ...

    def on_book_skip(self, book_id: str, book_name: str, output_path: Path) -> None:
        """Signal that a book was skipped because its output already exists."""
        ...

    def on_book_error(self, book_id: str, source_name: str, error: Exception) -> None:
        """Signal that a book failed during processing."""
        ...

    def on_batch_finish(self) -> None:
        """Signal completion of the active batch job."""
        ...


class NullProgressReporter:
    """No-op progress reporter used when callers do not attach a UI reporter."""

    def on_batch_start(self, total: int, *, label: str = "Converting") -> None:
        return None

    def on_book_start(self, book_id: str, book_name: str) -> None:
        return None

    def on_book_stage(self, event: ProgressEvent) -> None:
        return None

    def on_book_complete(self, book_id: str, book_name: str, output_path: Path) -> None:
        return None

    def on_book_skip(self, book_id: str, book_name: str, output_path: Path) -> None:
        return None

    def on_book_error(self, book_id: str, source_name: str, error: Exception) -> None:
        return None

    def on_batch_finish(self) -> None:
        return None


class RichProgressReporter:
    """Thread-safe Rich progress reporter with TTY Live display and non-TTY streaming."""

    def __init__(
        self,
        console: Console,
        *,
        is_batch: bool = False,
        show_live: bool | None = None,
        complete_prefix: str = "Converted",
    ) -> None:
        self._console = console
        self._is_batch = is_batch
        self._live_enabled = console.is_terminal if show_live is None else show_live
        self._complete_prefix = complete_prefix
        self._lock = threading.Lock()
        self._batch_total = 0
        self._succeeded = 0
        self._skipped = 0
        self._failed = 0
        self._overall_task_id: TaskID | None = None
        self._active_tasks: dict[str, TaskID] = {}

        if self._live_enabled:
            self._overall_progress: Progress | None = Progress(
                TextColumn("[bold]{task.description}[/bold]"),
                BarColumn(),
                TaskProgressColumn(),
                TextColumn("{task.fields[counts]}"),
                TimeElapsedColumn(),
                TextColumn("ETA"),
                TimeRemainingColumn(),
                console=console,
            )
            self._worker_progress: Progress | None = Progress(
                TextColumn("  "),
                SpinnerColumn(),
                TextColumn("{task.description}"),
                TextColumn("[dim]{task.fields[detail]}[/dim]"),
                TimeElapsedColumn(),
                console=console,
            )
            self._live: Live | None = Live(
                Group(self._overall_progress, self._worker_progress),
                console=console,
                refresh_per_second=10,
                transient=not is_batch,
            )
        else:
            self._overall_progress = None
            self._worker_progress = None
            self._live = None

    def __enter__(self) -> RichProgressReporter:
        if self._live is not None:
            self._live.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        with self._lock:
            for book_id in list(self._active_tasks):
                self._remove_worker_task_locked(book_id)
            should_make_transient = exc_type is not None or self._overall_task_id is None
        if self._live is not None:
            if should_make_transient:
                self._live.transient = True
            self._live.__exit__(exc_type, exc_val, exc_tb)

    def _format_counts(self) -> str:
        done = self._succeeded + self._skipped + self._failed
        return (
            f"{done}/{self._batch_total} books "
            f"({self._succeeded} ok, {self._skipped} skip, {self._failed} fail)"
        )

    def _remove_worker_task_locked(self, book_id: str) -> None:
        task_id = self._active_tasks.pop(book_id, None)
        if task_id is not None and self._worker_progress is not None:
            self._worker_progress.remove_task(task_id)

    def _advance_overall_locked(self) -> None:
        if self._overall_progress is not None and self._overall_task_id is not None:
            done = self._succeeded + self._skipped + self._failed
            self._overall_progress.update(
                self._overall_task_id,
                completed=done,
                counts=rich_escape(self._format_counts()),
            )

    def on_batch_start(self, total: int, *, label: str = "Converting") -> None:
        with self._lock:
            self._batch_total = total
            self._succeeded = 0
            self._skipped = 0
            self._failed = 0
            if self._overall_progress is not None:
                if self._overall_task_id is not None:
                    self._overall_progress.remove_task(self._overall_task_id)
                self._overall_task_id = self._overall_progress.add_task(
                    rich_escape(label),
                    total=total,
                    counts=rich_escape(self._format_counts()),
                )

    def on_book_start(self, book_id: str, book_name: str) -> None:
        with self._lock:
            if self._worker_progress is not None:
                self._remove_worker_task_locked(book_id)
                task_id = self._worker_progress.add_task(
                    rich_escape(book_name),
                    total=None,
                    detail="",
                )
                self._active_tasks[book_id] = task_id

    def on_book_stage(self, event: ProgressEvent) -> None:
        with self._lock:
            if self._worker_progress is not None:
                task_id = self._active_tasks.get(event.book_id)
                if task_id is None:
                    task_id = self._worker_progress.add_task(
                        rich_escape(event.book_name),
                        total=None,
                        detail="",
                    )
                    self._active_tasks[event.book_id] = task_id
                self._worker_progress.update(
                    task_id,
                    description=rich_escape(event.book_name),
                    detail=f"— {rich_escape(event.format_detail())}",
                )

    def on_book_complete(self, book_id: str, book_name: str, output_path: Path) -> None:
        with self._lock:
            self._remove_worker_task_locked(book_id)
            if self._is_batch:
                self._succeeded += 1
                self._advance_overall_locked()
                if not self._live_enabled:
                    self._console.print(
                        f"[green]{rich_escape(self._complete_prefix)}:[/green] "
                        f"{rich_escape(str(output_path))}",
                        soft_wrap=True,
                    )

    def on_book_skip(self, book_id: str, book_name: str, output_path: Path) -> None:
        with self._lock:
            self._remove_worker_task_locked(book_id)
            if self._is_batch:
                self._skipped += 1
                self._advance_overall_locked()
                if not self._live_enabled:
                    self._console.print(
                        f"[yellow]Skipped (already exists):[/yellow] "
                        f"{rich_escape(str(output_path))}",
                        soft_wrap=True,
                    )

    def on_book_error(self, book_id: str, source_name: str, error: Exception) -> None:
        with self._lock:
            self._remove_worker_task_locked(book_id)
            if self._is_batch:
                self._failed += 1
                self._advance_overall_locked()
                self._console.print(
                    f"[red]Failed:[/red] {rich_escape(source_name)} - "
                    f"{rich_escape(str(error))}",
                    soft_wrap=True,
                )

    def on_batch_finish(self) -> None:
        with self._lock:
            for book_id in list(self._active_tasks):
                self._remove_worker_task_locked(book_id)
