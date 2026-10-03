//! The elementwise slice operations of `Lanes`: each store, integer
//! conversion, and reduction gives the result of its `_with` method in the
//! default mode, for each view, lane count, and kind of vector of
//! `floaty_verify::kernels`. A kernel of a view gives the result of its
//! `_with` method. A store into cells that its view reads gives the store
//! into another slice. `convert_slice` of host values gives the conversion
//! of the values of the lane type.

use core::cell::Cell;
use std::panic::{AssertUnwindSafe, catch_unwind};

use floaty::elementwise::{
    Abs, Difference, Maximum, MaximumNumber, Minimum, MinimumNumber, Product, Quotient,
    RoundToIntegral, Splat, Sum,
};
use floaty::env::Mode;
use floaty::mode::{Ieee, X86Sse};
use floaty::{BF16, Binary, F16, F32, Float, Lanes, Rounding, ToInt, Vector};
use floaty_verify::encodings::Layout;
use floaty_verify::kernels::{LENGTHS, Mix, codes, integral_vector, pair};
use floaty_verify::random::SplitMix64;

/// The binary32 lanes with the default mode `M`.
type Single<M, const N: usize> = Lanes<Float<Binary<8>, 32, M>, N>;

/// The rounding directions of `RoundToIntegral`.
const DIRECTIONS: [Rounding; 8] = [
    Rounding::TiesToEven,
    Rounding::TiesToAway,
    Rounding::TiesTowardZero,
    Rounding::TowardPositive,
    Rounding::TowardNegative,
    Rounding::TowardZero,
    Rounding::AwayFromZero,
    Rounding::ToOdd,
];

/// Checks every operation of `N` lanes in the mode `M` on the view `view`
/// of `count` values, and returns the encodings of its store.
fn check<M: Mode, const N: usize>(
    view: impl Vector,
    count: usize,
    context: &dyn Fn() -> String,
) -> Vec<u32> {
    let mode = M::default();
    let zero = Float::<Binary<8>, 32, M>::from_bits(0);
    let (mut ours, mut engine) = (vec![zero; count], vec![zero; count]);
    Single::<M, N>::store(view, &mut ours[..]);
    Single::<M, N>::store_with(view, &mut engine[..], mode);
    let encodings: Vec<u32> = ours.iter().map(|value| value.to_bits()).collect();
    let engine_encodings: Vec<u32> = engine.iter().map(|value| value.to_bits()).collect();
    assert_eq!(encodings, engine_encodings, "store N = {N} {}", context());
    let mut hosts = vec![0.0_f32; count];
    Single::<M, N>::store(view, &mut hosts[..]);
    let host_encodings: Vec<u32> = hosts.iter().map(|value| value.to_bits()).collect();
    assert_eq!(host_encodings, encodings, "store f32 N = {N} {}", context());
    check_integers::<M, N, i8>(view, count, context);
    check_integers::<M, N, u8>(view, count, context);
    check_integers::<M, N, i32>(view, count, context);
    let reductions = [
        (
            "minimum_of",
            Single::<M, N>::minimum_of(view),
            Single::<M, N>::minimum_of_with(view, mode).0,
        ),
        (
            "maximum_of",
            Single::<M, N>::maximum_of(view),
            Single::<M, N>::maximum_of_with(view, mode).0,
        ),
        (
            "minimum_number_of",
            Single::<M, N>::minimum_number_of(view),
            Single::<M, N>::minimum_number_of_with(view, mode).0,
        ),
        (
            "maximum_number_of",
            Single::<M, N>::maximum_number_of(view),
            Single::<M, N>::maximum_number_of_with(view, mode).0,
        ),
        (
            "sum",
            Single::<M, N>::sum(view),
            Single::<M, N>::sum_with(view, mode).0,
        ),
        (
            "dot",
            Single::<M, N>::dot(view, view),
            Single::<M, N>::dot_with(view, view, mode).0,
        ),
    ];
    for (name, ours, engine) in reductions {
        assert_eq!(
            ours.to_bits(),
            engine.to_bits(),
            "{name} N = {N} {}",
            context()
        );
    }
    encodings
}

/// Checks `to_int_slice` into the integer type `I` against its `_with`
/// method.
fn check_integers<
    M: Mode,
    const N: usize,
    I: floaty::Integer + Copy + PartialEq + core::fmt::Debug,
>(
    view: impl Vector,
    count: usize,
    context: &dyn Fn() -> String,
) {
    let (mut ours, mut engine) = (vec![ToInt::<I>::Nan; count], vec![ToInt::<I>::Nan; count]);
    Single::<M, N>::to_int_slice(view, &mut ours);
    Single::<M, N>::to_int_slice_with(view, &mut engine, M::default());
    assert_eq!(
        ours,
        engine,
        "to_int_slice {} N = {N} {}",
        core::any::type_name::<I>(),
        context()
    );
}

/// Checks every operation in each lane count and both modes on a view, and
/// returns the encodings of the stores.
fn check_lanes(view: impl Vector, count: usize, context: &dyn Fn() -> String) -> Vec<u32> {
    let mut encodings = Vec::new();
    encodings.extend(check::<Ieee, 1>(view, count, context));
    encodings.extend(check::<Ieee, 2>(view, count, context));
    encodings.extend(check::<Ieee, 4>(view, count, context));
    encodings.extend(check::<Ieee, 8>(view, count, context));
    encodings.extend(check::<Ieee, 32>(view, count, context));
    encodings.extend(check::<X86Sse, 8>(view, count, context));
    encodings
}

/// Returns the encodings as binary32 values.
fn singles(encodings: &[u128]) -> Vec<F32> {
    encodings
        .iter()
        .map(|&bits| F32::from_bits(u32::try_from(bits).expect("a binary32 encoding")))
        .collect()
}

/// Checks each view of two vectors on `x` and `y` of `count` values.
fn check_views(x: impl Vector, y: impl Vector, count: usize, context: &dyn Fn() -> String) {
    let with = |name: &'static str| move || format!("{name} {}", context());
    check_lanes(Sum(x, y), count, &with("Sum"));
    check_lanes(Difference(x, y), count, &with("Difference"));
    check_lanes(Product(x, y), count, &with("Product"));
    check_lanes(Quotient(x, y), count, &with("Quotient"));
    check_lanes(Minimum(x, y), count, &with("Minimum"));
    check_lanes(Maximum(x, y), count, &with("Maximum"));
    check_lanes(MinimumNumber(x, y), count, &with("MinimumNumber"));
    check_lanes(MaximumNumber(x, y), count, &with("MaximumNumber"));
    check_lanes(Abs(x), count, &with("Abs"));
}

#[test]
fn views_of_two_vectors_give_the_default_mode_results() {
    let mut random = SplitMix64::new(0x656C_656D);
    for mix in Mix::ALL {
        for length in LENGTHS {
            let (a, b) = pair(Layout::BINARY32, mix, length, &mut random);
            let (x, y) = (singles(&a), singles(&b));
            let context = || format!("{mix:?} {length}");
            check_views(&x[..], &y[..], length, &context);
            let hosts: Vec<f32> = x.iter().map(|&value| f32::from(value)).collect();
            check_views(&hosts[..], &y[..], length, &|| format!("f32 {}", context()));
        }
    }
}

#[test]
fn rounding_gives_the_default_mode_results() {
    let mut random = SplitMix64::new(0x726E_6468);
    for length in LENGTHS {
        let x: Vec<F32> = integral_vector(length, &mut random)
            .into_iter()
            .map(F32::from_bits)
            .collect();
        let scale = Splat::new(F32::from_bits(0x437F_0000), length);
        for rounding in DIRECTIONS {
            let context = || format!("{rounding:?} {length}");
            check_lanes(
                RoundToIntegral {
                    values: &x[..],
                    rounding,
                },
                length,
                &context,
            );
            check_lanes(
                RoundToIntegral {
                    values: Product(&x[..], scale),
                    rounding,
                },
                length,
                &|| format!("scaled {}", context()),
            );
        }
    }
}

#[test]
fn integral_rounding_preserves_signs_and_magnitudes() {
    let magnitudes = [
        0,
        1,
        0x007F_FFFF,
        0x0080_0000,
        0x3E80_0000,
        0x3EFF_FFFF,
        0x3F00_0000,
        0x3F00_0001,
        0x3F40_0000,
        0x3F80_0000,
        0x3FC0_0000,
        0x4AFF_FFFF,
        0x4B00_0000,
        0x4B00_0001,
        0x7F7F_FFFF,
        0x7F80_0000,
    ];
    let values: [F32; 32] = core::array::from_fn(|index| {
        let sign = if index % 2 == 0 { 0 } else { 0x8000_0000 };
        F32::from_bits(magnitudes[index / 2] | sign)
    });
    for rounding in DIRECTIONS {
        let _ = check::<Ieee, 32>(
            RoundToIntegral {
                values: &values[..],
                rounding,
            },
            values.len(),
            &|| format!("{rounding:?} signed boundaries"),
        );
    }
}

#[test]
fn nested_views_and_narrow_operands_give_the_default_mode_results() {
    let mut random = SplitMix64::new(0x6E73_7464);
    for mix in Mix::ALL {
        for length in LENGTHS {
            let context = || format!("{mix:?} {length}");
            let (first, second) = pair(Layout::BINARY32, mix, length, &mut random);
            let (x, y) = (singles(&first), singles(&second));
            let (narrow_bits, half_bits) = pair(Layout::BFLOAT16, mix, length, &mut random);
            let narrow: Vec<BF16> = narrow_bits
                .iter()
                .map(|&bits| BF16::from_bits(u16::try_from(bits).expect("a bfloat16 encoding")))
                .collect();
            let half: Vec<F16> = half_bits
                .iter()
                .map(|&bits| F16::from_bits(u16::try_from(bits).expect("a binary16 encoding")))
                .collect();
            let unsigned = codes(length, &mut random);
            let (keep, eta) = (F32::from_bits(0x3F7F_0000), F32::from_bits(0x3B80_0000));
            check_lanes(
                Sum(
                    Product(&x[..], Splat::new(keep, length)),
                    Product(&y[..], Splat::new(eta, length)),
                ),
                length,
                &|| format!("update {}", context()),
            );
            check_lanes(
                Sum(
                    &x[..],
                    Product(
                        Quotient(
                            &unsigned[..],
                            Splat::new(F32::from_bits(0x437F_0000), length),
                        ),
                        Difference(&y[..], &x[..]),
                    ),
                ),
                length,
                &|| format!("reconstruction {}", context()),
            );
            check_lanes(
                MaximumNumber(
                    Minimum(&narrow[..], &half[..]),
                    Splat::new(
                        narrow.first().copied().unwrap_or(BF16::from_bits(0)),
                        length,
                    ),
                ),
                length,
                &|| format!("narrow {}", context()),
            );
        }
    }
}

#[test]
fn stores_into_cells_read_each_value_first() {
    let mut random = SplitMix64::new(0x6365_6C6C);
    for mix in Mix::ALL {
        for length in LENGTHS {
            let (a, b) = pair(Layout::BINARY32, mix, length, &mut random);
            let (x, y) = (singles(&a), singles(&b));
            let mut expected = vec![F32::from_bits(0); length];
            Single::<Ieee, 8>::store(MinimumNumber(&x[..], &y[..]), &mut expected[..]);
            let mut values = x.clone();
            let cells = Cell::from_mut(&mut values[..]).as_slice_of_cells();
            Single::<Ieee, 8>::store(MinimumNumber(cells, &y[..]), cells);
            let encodings = |values: &[F32]| -> Vec<u32> {
                values.iter().map(|value| value.to_bits()).collect()
            };
            assert_eq!(encodings(&values), encodings(&expected), "{mix:?} {length}");
            let mut hosts: Vec<f32> = x.iter().map(|&value| f32::from(value)).collect();
            let cells = Cell::from_mut(&mut hosts[..]).as_slice_of_cells();
            Single::<Ieee, 8>::store(MinimumNumber(cells, &y[..]), cells);
            let host_encodings: Vec<u32> = hosts.iter().map(|value| value.to_bits()).collect();
            assert_eq!(host_encodings, encodings(&expected), "f32 {mix:?} {length}");
        }
    }
}

/// Checks both overlap directions before a store can change any cell.
fn check_shifted_cell_stores<T: Copy + From<F32>, const N: usize>()
where
    F32: From<T>,
    for<'a> &'a [Cell<T>]: Vector,
{
    for count in [2, 3, 7, 8, 9, 31, 32, 33, 65] {
        let cells: Vec<_> = (0..=count)
            .map(|index| {
                let value = u32::try_from(index + 1).expect("the test index is at most 66");
                Cell::new(T::from(F32::from_int(value)))
            })
            .collect();
        let encodings = || {
            cells
                .iter()
                .map(|cell| F32::from(cell.get()).to_bits())
                .collect::<Vec<_>>()
        };
        let before = encodings();
        for (source_start, destination_start) in [(0, 1), (1, 0)] {
            let source = &cells[source_start..source_start + count];
            let destination = &cells[destination_start..destination_start + count];
            let view = RoundToIntegral {
                values: Abs(Sum(
                    destination,
                    Product(source, Splat::new(F32::from_bits(0x4000_0000), count)),
                )),
                rounding: Rounding::TiesToEven,
            };
            let result = catch_unwind(AssertUnwindSafe(|| {
                Single::<Ieee, N>::store(view, destination);
            }));
            assert!(result.is_err(), "shifted store: {N} lanes, {count} values");
            assert_eq!(encodings(), before, "a rejected store must not write");
            for rounding in [Rounding::TiesToEven, Rounding::TowardPositive] {
                let result = catch_unwind(AssertUnwindSafe(|| {
                    Single::<Ieee, N>::store_with(view, destination, rounding)
                }));
                assert!(
                    result.is_err(),
                    "shifted store_with: {N} lanes, {count} values, {rounding:?}"
                );
                assert_eq!(encodings(), before, "a rejected store must not write");
            }
        }
    }
}

#[test]
fn shifted_cell_stores_panic_before_writing() {
    check_shifted_cell_stores::<F32, 1>();
    check_shifted_cell_stores::<F32, 4>();
    check_shifted_cell_stores::<F32, 8>();
    check_shifted_cell_stores::<F32, 32>();
    check_shifted_cell_stores::<f32, 1>();
    check_shifted_cell_stores::<f32, 4>();
    check_shifted_cell_stores::<f32, 8>();
    check_shifted_cell_stores::<f32, 32>();
}

#[test]
fn adjacent_single_cell_stores_are_disjoint() {
    let one = 0x3F80_0000;
    let two = 0x4000_0000;
    let three = 0x4040_0000;
    for (source_start, destination_start, expected) in [(0, 1, [one, three]), (1, 0, [three, two])]
    {
        for with_flags in [false, true] {
            let cells = [one, two].map(|bits| Cell::new(F32::from_bits(bits)));
            let source = &cells[source_start..=source_start];
            let destination = &cells[destination_start..=destination_start];
            let view = Sum(source, destination);
            if with_flags {
                let flags = Single::<Ieee, 1>::store_with(view, destination, Rounding::TiesToEven);
                assert_eq!(flags, floaty::Flags::NONE);
            } else {
                Single::<Ieee, 1>::store(view, destination);
            }
            assert_eq!(cells.each_ref().map(|cell| cell.get().to_bits()), expected);
        }
    }
}

#[test]
fn conversions_of_host_values_give_the_conversions_of_the_values() {
    let mut random = SplitMix64::new(0x636E_7668);
    for mix in Mix::ALL {
        for length in LENGTHS {
            let (a, _) = pair(Layout::BINARY32, mix, length, &mut random);
            let x = singles(&a);
            let hosts: Vec<f32> = x.iter().map(|&value| f32::from(value)).collect();
            let (mut typed, mut host) = (
                vec![BF16::from_bits(0); length],
                vec![BF16::from_bits(0); length],
            );
            Lanes::<F32, 8>::convert_slice(&x, &mut typed);
            Lanes::<F32, 8>::convert_slice(&hosts, &mut host);
            let narrow_bits = |values: &[BF16]| -> Vec<u16> {
                values.iter().map(|value| value.to_bits()).collect()
            };
            assert_eq!(
                narrow_bits(&typed),
                narrow_bits(&host),
                "bfloat16 {mix:?} {length}"
            );
            let (mut typed, mut host) = (
                vec![F16::from_bits(0); length],
                vec![F16::from_bits(0); length],
            );
            Lanes::<F32, 8>::convert_slice(&x, &mut typed);
            Lanes::<F32, 8>::convert_slice(&hosts, &mut host);
            let half_bits = |values: &[F16]| -> Vec<u16> {
                values.iter().map(|value| value.to_bits()).collect()
            };
            assert_eq!(
                half_bits(&typed),
                half_bits(&host),
                "binary16 {mix:?} {length}"
            );
            let mut flagged = vec![F16::from_bits(0); length];
            Lanes::<F32, 8>::convert_slice_with(&hosts, &mut flagged, Ieee);
            assert_eq!(
                half_bits(&flagged),
                half_bits(&host),
                "binary16 with {mix:?} {length}"
            );
        }
    }
}
