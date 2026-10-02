//! The slice kernels of `Lanes` and the slice conversion: each entry point
//! gives the result of its `_with` method in the default mode, for each
//! kind of vector, lane count, and length of `floaty_verify::kernels`. Each
//! view of one vector, `&[F32]`, `&[f32]`, and `LittleEndian`, gives one
//! result. The mode `X86Sse` gives another NaN than `Ieee`, which a NaN sum
//! takes from the engine.

use floaty::env::Mode;
use floaty::mode::{Ieee, X86Sse};
use floaty::{BF16, Binary, F16, F32, F64, Float, Lanes, LittleEndian, ScaledCodes, Vector};
use floaty_verify::encodings::{Layout, boundary_encodings_u128, sample_encodings_u128};
use floaty_verify::kernels::{LENGTHS, Mix, codes, pair};
use floaty_verify::random::SplitMix64;

/// The binary32 lanes with the default mode `M`.
type Single<M, const N: usize> = Lanes<Float<Binary<8>, 32, M>, N>;

/// Checks every kernel with `N` lanes in the mode `M` on `x` and `y`, and
/// returns the bits of each result, so that two views of one vector can be
/// compared.
fn check<M: Mode, const N: usize>(
    x: impl Vector,
    y: impl Vector,
    context: &dyn Fn() -> String,
) -> [u32; 9] {
    let mode = M::default();
    let results = [
        (
            "sum",
            Single::<M, N>::sum(x),
            Single::<M, N>::sum_with(x, mode).0,
        ),
        (
            "dot",
            Single::<M, N>::dot(x, y),
            Single::<M, N>::dot_with(x, y, mode).0,
        ),
        (
            "dot_fused",
            Single::<M, N>::dot_fused(x, y),
            Single::<M, N>::dot_fused_with(x, y, mode).0,
        ),
        (
            "distance_square",
            Single::<M, N>::distance_square(x, y),
            Single::<M, N>::distance_square_with(x, y, mode).0,
        ),
        (
            "distance_square_fused",
            Single::<M, N>::distance_square_fused(x, y),
            Single::<M, N>::distance_square_fused_with(x, y, mode).0,
        ),
        (
            "norm",
            Single::<M, N>::norm(x),
            Single::<M, N>::norm_with(x, mode).0,
        ),
        (
            "norm_fused",
            Single::<M, N>::norm_fused(x),
            Single::<M, N>::norm_fused_with(x, mode).0,
        ),
        (
            "norm y",
            Single::<M, N>::norm(y),
            Single::<M, N>::norm_with(y, mode).0,
        ),
        (
            "sum y",
            Single::<M, N>::sum(y),
            Single::<M, N>::sum_with(y, mode).0,
        ),
    ];
    results.map(|(name, ours, engine)| {
        assert_eq!(
            ours.to_bits(),
            engine.to_bits(),
            "{name} N = {N} {}",
            context()
        );
        ours.to_bits()
    })
}

/// Checks every kernel in each lane count and both modes on `x` and `y`,
/// and returns the bits of each result.
fn check_lanes(x: impl Vector, y: impl Vector, context: &dyn Fn() -> String) -> Vec<u32> {
    let mut bits = Vec::new();
    bits.extend(check::<Ieee, 1>(x, y, context));
    bits.extend(check::<Ieee, 2>(x, y, context));
    bits.extend(check::<Ieee, 4>(x, y, context));
    bits.extend(check::<Ieee, 8>(x, y, context));
    bits.extend(check::<Ieee, 16>(x, y, context));
    bits.extend(check::<Ieee, 32>(x, y, context));
    bits.extend(check::<Ieee, 64>(x, y, context));
    bits.extend(check::<X86Sse, 8>(x, y, context));
    bits.extend(check::<X86Sse, 32>(x, y, context));
    bits
}

/// Returns the little-endian bytes of each encoding.
fn bytes<const B: usize>(encodings: &[u128]) -> Vec<u8> {
    encodings
        .iter()
        .flat_map(|&bits| {
            let le = bits.to_le_bytes();
            <[u8; B]>::try_from(&le[..B]).expect("an encoding of B bytes")
        })
        .collect()
}

/// Returns the encodings as values of a 32-bit format.
fn singles(encodings: &[u128]) -> Vec<F32> {
    encodings
        .iter()
        .map(|&bits| F32::from_bits(u32::try_from(bits).expect("a binary32 encoding")))
        .collect()
}

/// Returns the encodings as values of a 16-bit format.
fn halves<S: floaty::format::Standard<16, Bits = u16>>(encodings: &[u128]) -> Vec<Float<S, 16>> {
    encodings
        .iter()
        .map(|&bits| Float::from_bits(u16::try_from(bits).expect("a 16-bit encoding")))
        .collect()
}

#[test]
fn binary32_vectors_give_the_default_mode_results() {
    let mut random = SplitMix64::new(0x6B65_726E);
    for mix in Mix::ALL {
        for length in LENGTHS {
            let (a, b) = pair(Layout::BINARY32, mix, length, &mut random);
            let (x, y) = (singles(&a), singles(&b));
            let context = || format!("{mix:?} {length}");
            let typed = check_lanes(&x[..], &y[..], &context);
            let hosts: Vec<f32> = x.iter().map(|&value| f32::from(value)).collect();
            let host_y: Vec<f32> = y.iter().map(|&value| f32::from(value)).collect();
            assert_eq!(
                check_lanes(&hosts[..], &host_y[..], &context),
                typed,
                "&[f32] {}",
                context()
            );
            let (row_x, row_y) = (bytes::<4>(&a), bytes::<4>(&b));
            let little = check_lanes(
                LittleEndian::<F32>::new(&row_x),
                LittleEndian::<F32>::new(&row_y),
                &context,
            );
            assert_eq!(little, typed, "LittleEndian {}", context());
        }
    }
}

#[test]
fn bfloat16_and_binary16_vectors_give_the_default_mode_results() {
    let mut random = SplitMix64::new(0x6266_3136);
    for mix in Mix::ALL {
        for length in LENGTHS {
            let context = || format!("{mix:?} {length}");
            let (first, second) = pair(Layout::BFLOAT16, mix, length, &mut random);
            let (x, y): (Vec<BF16>, Vec<BF16>) = (halves(&first), halves(&second));
            let typed = check_lanes(&x[..], &y[..], &context);
            let (row_x, row_y) = (bytes::<2>(&first), bytes::<2>(&second));
            let little = check_lanes(
                LittleEndian::<BF16>::new(&row_x),
                LittleEndian::<BF16>::new(&row_y),
                &context,
            );
            assert_eq!(little, typed, "LittleEndian bfloat16 {}", context());

            let (first, second) = pair(Layout::BINARY16, mix, length, &mut random);
            let (x, y): (Vec<F16>, Vec<F16>) = (halves(&first), halves(&second));
            let typed = check_lanes(&x[..], &y[..], &context);
            let row_x = bytes::<2>(&first);
            let little = check_lanes(LittleEndian::<F16>::new(&row_x), &y[..], &context);
            assert_eq!(little, typed, "LittleEndian binary16 {}", context());
            // A mixed pair: binary16 against bfloat16.
            let bfloats: Vec<BF16> = halves(&second);
            check_lanes(&x[..], &bfloats[..], &context);
        }
    }
}

#[test]
fn integer_codes_give_the_default_mode_results() {
    let mut random = SplitMix64::new(0x636F_6465);
    let boundary = singles(&boundary_encodings_u128(Layout::BINARY32));
    // Each boundary encoding serves as a scale in turn: zeros, subnormals,
    // the largest values, infinities, and NaNs.
    let mut boundary_scales = boundary.iter().cycle();
    for mix in Mix::ALL {
        for length in LENGTHS {
            let context = || format!("{mix:?} {length}");
            let (_, b) = pair(Layout::BINARY32, mix, length, &mut random);
            let y = singles(&b);
            let unsigned = codes(length, &mut random);
            check_lanes(&unsigned[..], &y[..], &context);
            let signed: Vec<i8> = unsigned.iter().map(|&code| code.cast_signed()).collect();
            let boundary_scale = *boundary_scales.next().expect("the cycle never ends");
            for scale in [boundary_scale, F32::from_bits(0x3C00_0000)] {
                let scaled = ScaledCodes {
                    codes: &signed,
                    scale,
                };
                check_lanes(scaled, &y[..], &|| format!("{} scale {scale:?}", context()));
            }
        }
    }
}

/// Checks the rows of one matrix in `N` lanes: each row of `dot_rows` and
/// `distance_square_rows` is the kernel of the row, and the `_with` methods
/// give the same results.
fn check_rows<const N: usize>(rows: &[F32], query: &[F32], context: &dyn Fn() -> String) {
    let count = rows.len() / query.len().max(1);
    let count = if query.is_empty() { 3 } else { count };
    let rows = &rows[..count * query.len()];
    let (mut dots, mut dots_with) = (vec![0.0_f32; count], vec![F32::from_bits(0); count]);
    Lanes::<F32, N>::dot_rows(rows, query, &mut dots);
    let _ = Lanes::<F32, N>::dot_rows_with(rows, query, &mut dots_with, Ieee);
    let (mut distances, mut distances_with) =
        (vec![F32::from_bits(0); count], vec![0.0_f32; count]);
    Lanes::<F32, N>::distance_square_rows(rows, query, &mut distances);
    let _ = Lanes::<F32, N>::distance_square_rows_with(rows, query, &mut distances_with, Ieee);
    for index in 0..count {
        let row = &rows[index * query.len()..(index + 1) * query.len()];
        let dot = Lanes::<F32, N>::dot(row, query).to_bits();
        assert_eq!(
            dots[index].to_bits(),
            dot,
            "dot row {index} N = {N} {}",
            context()
        );
        assert_eq!(
            dots_with[index].to_bits(),
            dot,
            "dot_with row {index} {}",
            context()
        );
        let distance = Lanes::<F32, N>::distance_square(row, query).to_bits();
        assert_eq!(
            distances[index].to_bits(),
            distance,
            "distance row {index} {}",
            context()
        );
        assert_eq!(
            distances_with[index].to_bits(),
            distance,
            "distance_with row {index} {}",
            context()
        );
    }
}

#[test]
fn rows_give_the_kernel_of_each_row() {
    let mut random = SplitMix64::new(0x726F_7773);
    for mix in Mix::ALL {
        let (a, b) = pair(Layout::BINARY32, mix, 1027, &mut random);
        let (rows, query) = (singles(&a), singles(&b));
        for length in [0, 1, 7, 8, 9, 33, 100] {
            let context = || format!("{mix:?} query {length}");
            let query = &query[..length];
            check_rows::<1>(&rows, query, &context);
            check_rows::<8>(&rows, query, &context);
            check_rows::<32>(&rows, query, &context);
        }
    }
}

/// Rows of at most eight values take a path of their own. A row of -0
/// products sums to +0 in the lanes, and every count of rows leaves a
/// different count after the last group of four.
#[test]
fn short_rows_of_negative_zero_products_give_positive_zero() {
    let negative_zero = F32::from_bits(0x8000_0000);
    let one = F32::from_bits(0x3F80_0000);
    for length in 1..=8 {
        for count in 1..=9 {
            let rows = vec![negative_zero; length * count];
            let query = vec![one; length];
            let context = || format!("query {length} rows {count}");
            check_rows::<8>(&rows, &query, &context);
            check_rows::<32>(&rows, &query, &context);
        }
    }
}

/// Checks `convert_slice` with `$lanes` lanes from `$from` to `$to` on
/// `$values`: each value gives the conversion of its own value, and
/// `convert_slice_with` gives the same values.
macro_rules! check_conversion {
    ($from:ty, $to:ty, $values:expr, $lanes:literal) => {{
        let values: &[$from] = $values;
        let zero = <$to>::from_bits(0);
        let (mut ours, mut engine) = (vec![zero; values.len()], vec![zero; values.len()]);
        Lanes::<$from, $lanes>::convert_slice(values, &mut ours);
        let _ = Lanes::<$from, $lanes>::convert_slice_with(values, &mut engine, <$to>::ENV);
        for ((value, ours), engine) in values.iter().zip(&ours).zip(&engine) {
            let expected = value.convert_with::<$to>(<$to>::ENV).0.to_bits();
            let context = || format!("{value:?} to {} N = {}", stringify!($to), $lanes);
            assert_eq!(ours.to_bits(), expected, "{}", context());
            assert_eq!(engine.to_bits(), expected, "with {}", context());
        }
    }};
}

/// Checks the conversions from `$from` to `$to` in each lane count.
macro_rules! check_conversions {
    ($from:ty, $to:ty, $values:expr) => {{
        let values = $values;
        for length in [0, 1, 7, 8, 9, 31, 32, 33, values.len()] {
            check_conversion!($from, $to, &values[..length], 1);
            check_conversion!($from, $to, &values[..length], 8);
            check_conversion!($from, $to, &values[..length], 32);
        }
    }};
}

#[test]
fn slice_conversions_give_the_conversion_of_each_value() {
    let mut random = SplitMix64::new(0x636F_6E76);
    let singles = singles(&sample_encodings_u128(
        Layout::BINARY32,
        20_000,
        &mut random,
    ));
    check_conversions!(F32, BF16, &singles);
    check_conversions!(F32, F16, &singles);
    check_conversions!(F32, F64, &singles);
    let bfloats: Vec<BF16> = halves(&sample_encodings_u128(
        Layout::BFLOAT16,
        20_000,
        &mut random,
    ));
    check_conversions!(BF16, F32, &bfloats);
    let binary16: Vec<F16> = halves(&sample_encodings_u128(
        Layout::BINARY16,
        20_000,
        &mut random,
    ));
    check_conversions!(F16, F32, &binary16);
}

/// The binary32 encodings with every high half and the low halves around
/// each rounding case of binary16: the ties at bits 12 to 15, and the bits
/// just above and below them. The high halves hold the ties of the
/// subnormal results, the overflow to the infinity, and the NaNs.
#[test]
fn binary16_slice_conversion_rounds_each_case() {
    const LOWS: [u32; 22] = [
        0x0000, 0x1000, 0x2000, 0x3000, 0x4000, 0x5000, 0x6000, 0x7000, 0x8000, 0x9000, 0xA000,
        0xB000, 0xC000, 0xD000, 0xE000, 0xF000, 0x0001, 0x0FFF, 0x1001, 0x7FFF, 0x8001, 0xFFFF,
    ];
    let singles: Vec<F32> = (0..=u32::from(u16::MAX))
        .flat_map(|high| LOWS.map(|low| F32::from_bits((high << 16) | low)))
        .collect();
    check_conversion!(F32, F16, &singles, 8);
    check_conversion!(F32, F16, &singles, 32);
}
