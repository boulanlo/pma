use itertools::Itertools;
use pma::pma::PMA;

fn main() {
    let size = 2000u32;

    let a = (0u32..10000).map(|x| 2 * x);

    let mut pma = PMA::from_iterator(a, 0.3..0.7, 0.08..0.92, 8);

    for chunk in &(0u32..size).map(|x| (2 * x) + 1).chunks(100) {
        pma.insert_bulk(chunk.collect::<Vec<u32>>());
    }

    let results = pma.elements().collect::<Vec<&u32>>();

    assert_eq!(results.len(), 12000);

    println!("{:?}", results);

    for x in results.as_slice().windows(2) {
        assert!(x[0] <= x[1]);
    }
}
