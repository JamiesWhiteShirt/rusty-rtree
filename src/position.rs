pub trait Positioned<N, const D: usize> {
    fn position(&self) -> Position<N, D>;
}

#[derive(Clone, Copy)]
pub struct Position<N, const D: usize>([N; D]);
