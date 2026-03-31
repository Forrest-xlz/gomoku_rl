class PlayoutLadder:
    def __init__(
        self,
        playouts: list[int],
        win_rate_threshold: float = 0.70,
        index: int = 0,
        history: list[tuple[int, float]] | None = None,
    ) -> None:
        if not playouts:
            raise ValueError("playouts must not be empty")
        if sorted(playouts) != playouts:
            raise ValueError("playouts must be sorted in ascending order")

        self.playouts = list(playouts)
        self.win_rate_threshold = float(win_rate_threshold)
        self.index = int(index)
        self.history = [] if history is None else list(history)

    def current(self) -> int:
        return int(self.playouts[self.index])

    def exhausted(self) -> bool:
        return self.index >= len(self.playouts) - 1

    def update(self, win_rate: float) -> bool:
        self.history.append((self.current(), float(win_rate)))
        if self.exhausted():
            return False
        if win_rate >= self.win_rate_threshold:
            self.index += 1
            return True
        return False