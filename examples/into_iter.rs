use pma::PMA;

fn main() {
    let pma = PMA::from_iterator(0u32..2048, 0.3..0.7, 0.08..0.92, 8);

    println!("results: {:?}", pma.iter_chunks(0..512).count())
}
