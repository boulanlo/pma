use pma::merge::Merge;

fn main() {
    let a = (0..5).peekable();
    let b = (2..8).peekable();

    println!("{:?}", Merge::new(a, b).collect::<Vec<usize>>());
}
