use pma::pma::PMA;
use rand::seq::SliceRandom;

fn random_vec(size: usize) -> Vec<usize> {
    let mut v = (0..size).collect::<Vec<_>>();
    v.shuffle(&mut rand::thread_rng());
    v
}

fn main() {
    let vector = random_vec(5000);
    let mut iterator = vector.into_iter();

    let mut pma = PMA::from_iterator(
        std::iter::once(iterator.next().unwrap()),
        0.3..0.7,
        0.08..0.92,
        8,
    );

    for i in iterator {
        pma.insert(i);
    }
}
