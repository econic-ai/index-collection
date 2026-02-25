use radix::{BuildError, IndexTable, M0Table, RadixConfig};

fn main() -> Result<(), BuildError> {
    let mut table = M0Table::new(RadixConfig {
        prefix_bits: 10,
        arena_capacity: 1 << 20,
    })?;

    let id = 0xfeed_face_cafe_beefu64;
    table.insert(id);

    println!(
        "M0 table ready. contains(id)={}, load_factor={:.6}",
        table.contains(id),
        table.load_factor()
    );

    Ok(())
}
