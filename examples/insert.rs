use pma::pma::PMA;

fn main() {
    let mut pma = PMA::from_iterator(0u32..16, 0.3..0.7, 0.08..0.92, 8);

    for i in 0u32..20 {
        dbg!(i);
        pma.insert(i);
    }
    dbg!(&pma.data);
}
