use pma::PMA;

fn main() {
    let pma = PMA::from_iterator(0u32..21, 0.3..0.7, 0.08..0.92);

    eprintln!("{:?}", pma.elements_with_holes());
}

// n = nombre elems = 5
// P = taille vec
// t = ceil(log2 2n) = 4
// ceil((2n/t)).next_power_of_two -> 2^h = 4
// 2^h * t -> taille P = 20
