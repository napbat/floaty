//! The vectors that the slice kernels of `Lanes` read, and the conversion of
//! their values to binary32.

use core::marker::PhantomData;

use crate::env::{Behavior, Flags, Mode};
use crate::float::{F32, Float};
use crate::format::Binary;
use crate::host::{self, Isa, Load};
use crate::sealed::Sealed;

/// The binary32 type with the default mode `M`.
pub(super) type Single<M> = Float<Binary<8>, 32, M>;

/// A float type whose values the kernels of [`Lanes`](crate::Lanes) widen
/// to binary32: binary32, bfloat16, and binary16, in any mode. The widening
/// is exact. The trait is sealed.
pub trait Widen: Sealed + Copy {
    /// Returns the value as a binary32 value of the mode `M`, and the flags
    /// of the conversion.
    #[doc(hidden)]
    fn widen_with<M: Mode, B: Behavior>(self, behavior: B) -> (Single<M>, Flags);

    /// Returns the binary32 encodings of `values` from the host unit, in the
    /// instruction set `H`, or `None` when it has no instruction for the
    /// widening.
    #[doc(hidden)]
    fn widen_on_host<H: Isa, const N: usize>(values: &[Self; N]) -> Option<[u32; N]>;
}

impl<I: Mode> Widen for Float<Binary<8>, 32, I> {
    /// A binary32 value enters its lane as it is, so a signaling NaN stays
    /// signaling for the first operation that reads it.
    #[inline]
    fn widen_with<M: Mode, B: Behavior>(self, _behavior: B) -> (Single<M>, Flags) {
        (self.with_mode(), Flags::NONE)
    }

    #[inline]
    fn widen_on_host<H: Isa, const N: usize>(values: &[Self; N]) -> Option<[u32; N]> {
        Some(encodings(values, 0, Float::to_bits))
    }
}

impl<I: Mode> Widen for Float<Binary<8>, 16, I> {
    #[inline]
    fn widen_with<M: Mode, B: Behavior>(self, behavior: B) -> (Single<M>, Flags) {
        self.convert_with::<Single<M>>(behavior)
    }

    /// A bfloat16 encoding is the high half of a binary32 encoding.
    #[inline]
    fn widen_on_host<H: Isa, const N: usize>(values: &[Self; N]) -> Option<[u32; N]> {
        Some(encodings(values, 0, |value| {
            u32::from(value.to_bits()) << 16
        }))
    }
}

impl<I: Mode> Widen for Float<Binary<5>, 16, I> {
    #[inline]
    fn widen_with<M: Mode, B: Behavior>(self, behavior: B) -> (Single<M>, Flags) {
        self.convert_with::<Single<M>>(behavior)
    }

    #[inline]
    fn widen_on_host<H: Isa, const N: usize>(values: &[Self; N]) -> Option<[u32; N]> {
        host::packed::widen_halves::<H, N>(&encodings(values, 0, Float::to_bits))
    }
}

/// Returns the result of `encode` for each value. `fill` is any value of
/// the result type, which every lane holds before `encode` writes it.
#[inline]
fn encodings<T: Copy, R: Copy, const N: usize>(
    values: &[T; N],
    fill: R,
    encode: impl Fn(T) -> R,
) -> [R; N] {
    // A loop, not `array::map`, which LLVM calls out of line.
    let mut encodings = [fill; N];
    encodings
        .iter_mut()
        .zip(values)
        .for_each(|(encoding, &value)| *encoding = encode(value));
    encodings
}

/// A vector that the kernels of [`Lanes`](crate::Lanes) read. The trait is
/// sealed.
///
/// - `&[T]` of a [`Widen`] type: binary32, bfloat16, or binary16 values.
/// - `&[f32]`: host binary32 values, read by their bits, as `From` reads
///   them.
/// - `&[u8]`: unsigned integer codes, each converted to binary32 as
///   `from_int_with` converts it in the behavior of the kernel: exactly,
///   unless the behavior limits the precision.
/// - [`LittleEndian`]: the little-endian encodings of [`Widen`] values in
///   bytes, at any alignment.
/// - [`ScaledCodes`]: signed integer codes, each converted to binary32 as a
///   `&[u8]` code converts, and multiplied by a scale.
pub trait Vector: Sealed + Load {
    /// Returns value `index` as a binary32 value of the mode `M`, and the
    /// flags of its conversion.
    #[doc(hidden)]
    fn value_with<M: Mode, B: Behavior>(self, index: usize, behavior: B) -> (Single<M>, Flags);
}

/// Returns the binary32 encodings of the `N` values of `values` from
/// `start`, from `widen`. `values` holds at least `start + N` values.
#[inline]
fn load_full<E: Copy, const N: usize>(
    values: &[E],
    start: usize,
    widen: impl Fn(&[E; N]) -> Option<[u32; N]>,
) -> Option<[u32; N]> {
    let chunk: &[E; N] = values[start..start + N]
        .try_into()
        .expect("the caller loads a chunk inside the vector");
    widen(chunk)
}

/// Returns the binary32 encodings of the values of `values` from `start`,
/// from `widen`, with +0 in the lanes past the last value. `values` holds
/// more than `start` values.
#[inline]
fn load_rest<E: Copy, const N: usize>(
    values: &[E],
    start: usize,
    widen: impl Fn(&[E; N]) -> Option<[u32; N]>,
) -> Option<[u32; N]> {
    let rest = &values[start..];
    // A copy of a value fills the lanes past the end. Their encodings
    // become +0 below, so the value of the copy does not matter.
    let mut chunk = [rest[0]; N];
    chunk[..rest.len()].copy_from_slice(rest);
    let mut encodings = widen(&chunk)?;
    encodings[rest.len()..].fill(0);
    Some(encodings)
}

/// Implements [`Load`] for a slice of `$element`, whose chunks `$widen`
/// widens in the instruction set `H`.
macro_rules! load_slice {
    ($element:ty, |$chunk:ident| $widen:expr) => {
        #[inline]
        fn count(self) -> usize {
            self.len()
        }

        #[inline]
        fn load<H: Isa, const N: usize>(self, start: usize) -> Option<[u32; N]> {
            load_full::<$element, N>(self, start, |$chunk| $widen)
        }

        #[inline]
        fn load_rest<H: Isa, const N: usize>(self, start: usize) -> Option<[u32; N]> {
            load_rest::<$element, N>(self, start, |$chunk| $widen)
        }

        #[inline]
        fn part(self, start: usize, count: usize) -> Self {
            &self[start..start + count]
        }
    };
}

impl<T: Widen> Sealed for &[T] {}

impl<T: Widen> Load for &[T] {
    load_slice!(T, |chunk| T::widen_on_host::<H, N>(chunk));
}

impl<T: Widen> Vector for &[T] {
    #[inline]
    fn value_with<M: Mode, B: Behavior>(self, index: usize, behavior: B) -> (Single<M>, Flags) {
        self[index].widen_with(behavior)
    }
}

impl Sealed for &[f32] {}

impl Load for &[f32] {
    load_slice!(f32, |chunk| Some(encodings(chunk, 0, f32::to_bits)));
}

impl Vector for &[f32] {
    #[inline]
    fn value_with<M: Mode, B: Behavior>(self, index: usize, behavior: B) -> (Single<M>, Flags) {
        F32::from(self[index]).widen_with(behavior)
    }
}

impl Sealed for &[u8] {}

impl Load for &[u8] {
    load_slice!(u8, |chunk| host::packed::widen_codes::<H, N>(chunk));
}

impl Vector for &[u8] {
    #[inline]
    fn value_with<M: Mode, B: Behavior>(self, index: usize, behavior: B) -> (Single<M>, Flags) {
        Single::from_int_with(self[index], behavior)
    }
}

/// The little-endian encodings of values of a [`Widen`] type `T` in bytes,
/// as a stored row holds them, at any alignment.
///
/// ```
/// use floaty::{BF16, F32, Lanes, LittleEndian};
///
/// let row = [0x80, 0x3F, 0x00, 0x40]; // bfloat16 1 and 2
/// let query = [0x4040_0000, 0x4080_0000].map(F32::from_bits); // 3 and 4
/// let dot = Lanes::<F32, 8>::dot(LittleEndian::<BF16>::new(&row), &query[..]);
/// assert_eq!(dot.to_bits(), 0x4130_0000); // 11
/// ```
#[derive(Debug)]
pub struct LittleEndian<'a, T> {
    bytes: &'a [u8],
    marker: PhantomData<T>,
}

impl<T> Clone for LittleEndian<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for LittleEndian<'_, T> {}

impl<'a, T: Widen> LittleEndian<'a, T> {
    /// Makes a vector of the encodings in `bytes`, each `size_of::<T>()`
    /// bytes long.
    ///
    /// # Panics
    ///
    /// Panics when the length of `bytes` is not a multiple of the size of an
    /// encoding.
    #[must_use]
    pub fn new(bytes: &'a [u8]) -> Self {
        assert!(
            bytes.len().is_multiple_of(size_of::<T>()),
            "the bytes hold whole encodings"
        );
        Self {
            bytes,
            marker: PhantomData,
        }
    }
}

/// Implements [`Vector`] for the little-endian encodings of a format of
/// `$bytes` bytes, whose encoding is the integer type `$bits`.
macro_rules! little_endian {
    ($exponent:literal, $width:literal, $bits:ty, $bytes:literal) => {
        impl<I: Mode> Sealed for LittleEndian<'_, Float<Binary<$exponent>, $width, I>> {}

        impl<'a, I: Mode> LittleEndian<'a, Float<Binary<$exponent>, $width, I>> {
            /// Returns the encodings, each in `$bytes` bytes.
            #[inline]
            fn encodings(self) -> &'a [[u8; $bytes]] {
                self.bytes.as_chunks().0
            }

            /// Returns the value of an encoding.
            #[inline]
            fn value(encoding: [u8; $bytes]) -> Float<Binary<$exponent>, $width, I> {
                Float::from_bits(<$bits>::from_le_bytes(encoding))
            }

            /// Returns the binary32 encodings of the values of `chunk` from
            /// the host unit, in the instruction set `H`.
            #[inline]
            fn widen<H: Isa, const N: usize>(chunk: &[[u8; $bytes]; N]) -> Option<[u32; N]> {
                let values = encodings(chunk, Self::value([0; $bytes]), Self::value);
                Widen::widen_on_host::<H, N>(&values)
            }
        }

        impl<I: Mode> Load for LittleEndian<'_, Float<Binary<$exponent>, $width, I>> {
            #[inline]
            fn count(self) -> usize {
                self.bytes.len() / $bytes
            }

            #[inline]
            fn load<H: Isa, const N: usize>(self, start: usize) -> Option<[u32; N]> {
                load_full(self.encodings(), start, Self::widen::<H, N>)
            }

            #[inline]
            fn load_rest<H: Isa, const N: usize>(self, start: usize) -> Option<[u32; N]> {
                load_rest(self.encodings(), start, Self::widen::<H, N>)
            }

            #[inline]
            fn part(self, start: usize, count: usize) -> Self {
                Self {
                    bytes: &self.bytes[start * $bytes..(start + count) * $bytes],
                    marker: PhantomData,
                }
            }
        }

        impl<I: Mode> Vector for LittleEndian<'_, Float<Binary<$exponent>, $width, I>> {
            #[inline]
            fn value_with<M: Mode, B: Behavior>(
                self,
                index: usize,
                behavior: B,
            ) -> (Single<M>, Flags) {
                Self::value(self.encodings()[index]).widen_with(behavior)
            }
        }
    };
}

little_endian!(8, 32, u32, 4);
little_endian!(8, 16, u16, 2);
little_endian!(5, 16, u16, 2);

/// Signed integer codes and a binary32 scale. Value `i` is `codes[i]`,
/// converted to binary32 as a `&[u8]` code converts, times `scale`, rounded.
///
/// ```
/// use floaty::{F32, Lanes, ScaledCodes};
///
/// let half = F32::from_bits(0x3F00_0000);
/// let codes = ScaledCodes { codes: &[-4, 6], scale: half }; // -2 and 3
/// let one = [0x3F80_0000; 2].map(F32::from_bits);
/// assert_eq!(Lanes::<F32, 2>::dot(codes, &one[..]).to_bits(), 0x3F80_0000); // 1
/// ```
#[derive(Clone, Copy, Debug)]
pub struct ScaledCodes<'a> {
    /// The codes.
    pub codes: &'a [i8],
    /// The scale of every code.
    pub scale: F32,
}

impl Sealed for ScaledCodes<'_> {}

impl Load for ScaledCodes<'_> {
    #[inline]
    fn count(self) -> usize {
        self.codes.len()
    }

    #[inline]
    fn load<H: Isa, const N: usize>(self, start: usize) -> Option<[u32; N]> {
        let scale = self.scale.to_bits();
        load_full(self.codes, start, |chunk| {
            host::packed::widen_scaled_codes::<H, N>(chunk, scale)
        })
    }

    #[inline]
    fn load_rest<H: Isa, const N: usize>(self, start: usize) -> Option<[u32; N]> {
        let scale = self.scale.to_bits();
        load_rest(self.codes, start, |chunk| {
            host::packed::widen_scaled_codes::<H, N>(chunk, scale)
        })
    }

    #[inline]
    fn part(self, start: usize, count: usize) -> Self {
        Self {
            codes: &self.codes[start..start + count],
            scale: self.scale,
        }
    }
}

impl Vector for ScaledCodes<'_> {
    #[inline]
    fn value_with<M: Mode, B: Behavior>(self, index: usize, behavior: B) -> (Single<M>, Flags) {
        let (code, code_flags) = Single::<M>::from_int_with(self.codes[index], behavior);
        let (value, flags) = code.mul_with(self.scale.with_mode(), behavior);
        (value, code_flags | flags)
    }
}
