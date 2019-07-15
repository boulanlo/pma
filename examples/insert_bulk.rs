use pma::pma::PMA;

fn main() {
    let mut pma = PMA::from_iterator(0u32..16, 0.3..0.7, 0.08..0.92, 8);

    pma.insert_bulk(std::iter::repeat(1).take(15).collect());
}
