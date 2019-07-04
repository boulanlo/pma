use pma::PMA;

fn main() {
    for i in 1..2060 {
        let pma = PMA::from_iterator(0u32..i, 0.3..0.7, 0.08..0.92, 14);
        println!("{} {}", i, pma.density());
    }
}
