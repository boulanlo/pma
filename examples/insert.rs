use pma::pma::PMA;

fn main() {
    let mut pma = PMA::from_iterator(0u32..16, 0.3..0.7, 0.08..0.92, 8);

    for i in 16u32..32 {
        pma.insert(i);
    }
    //dbg!(pma.data);
    dbg!(&pma.elements().collect::<Vec<&u32>>());
}
