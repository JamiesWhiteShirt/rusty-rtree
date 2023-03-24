struct Line<N, const D: usize> {
    start: [N; D],
    end: [N; D],
}

impl<N: Ord + Copy, const D: usize> Bounded<N, D> for Line<N, D> {
    fn bounds(&self) -> Bounds<N, D> {
        let min = from_iter(
            self.start
                .into_iter()
                .zip(self.end)
                .map(|(start, end)| cmp::min(start, end)),
        )
        .unwrap();
        let max = from_iter(
            self.start
                .into_iter()
                .zip(self.end)
                .map(|(start, end)| cmp::max(start, end)),
        )
        .unwrap();
        Bounds { min, max }
    }
}
