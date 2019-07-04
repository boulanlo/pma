use pma::PMA;

fn main() {
    let pma = PMA::from_iterator(0u32..2048, 0.3..0.7, 0.08..0.92);

    let result: Vec<u32> = pma.into_iter().collect();
    let expected: Vec<u32> = (0u32..2048).collect();
    assert_eq!(result, expected);
}
