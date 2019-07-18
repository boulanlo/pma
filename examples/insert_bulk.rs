use pma::pma::PMA;

fn main() {
    let a = (0u32..2000).map(|x| 2 * x);
    let b = (0u32..10000).map(|x| (2 * x) + 1).collect::<Vec<u32>>();

    let mut pma = PMA::from_iterator(a, 0.3..0.7, 0.08..0.92, 8);

    pma.insert_bulk(b);

    let results = pma.elements().collect::<Vec<&u32>>();

    eprintln!("{:?}", results);
    for x in results.as_slice().windows(2) {
        assert!(x[0] <= x[1]);
    }
}
