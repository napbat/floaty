//! Format descriptions: the `Standard` trait, the `Binary<E, Enc>` family, the
//! encodings, and the table that maps each width to its storage type.

use core::fmt::Debug;
use core::hash::Hash;
use core::marker::PhantomData;

use crate::binary::{EncodingKind, Unpacked};
use crate::env::{Env, Flags};
use crate::exact::Unrounded;
use crate::float::Class;
use crate::limbs::Limbs;
use crate::sealed::Sealed;

use self::internal::LimbConversion;

/// A floating-point format family at a width of `W` bits.
///
/// A standard and a width together select a format, its storage type, and
/// its engine. The trait is sealed. Its hidden items are internal to the
/// crate and are not part of the API.
pub trait Standard<const W: usize>: Sealed + Sized + 'static {
    /// The integer type that stores an encoding.
    type Bits: Bits;

    /// The precision in bits, including the leading bit.
    const PRECISION: u32;

    /// The exponent of the largest finite value, `emax` in IEEE 754.
    const EMAX: i32;

    /// The exponent of the smallest normal value, `emin` in IEEE 754.
    const EMIN: i32;

    /// The number of NaN payload bits, below the quiet bit.
    #[doc(hidden)]
    const PAYLOAD_BITS: u32;

    /// Clears the storage bits above `W`.
    #[doc(hidden)]
    fn mask(bits: Self::Bits) -> Self::Bits;

    /// Decodes an encoding. The bits above `W` must be zero.
    #[doc(hidden)]
    fn unpack(bits: Self::Bits) -> Unpacked<<Self::Bits as LimbConversion>::Limbs>;

    /// Returns the class of an encoding. The bits above `W` must be zero.
    #[doc(hidden)]
    fn classify(bits: Self::Bits) -> Class;

    /// Returns `true` when the encoding is the canonical encoding of its value.
    /// The bits above `W` must be zero.
    #[doc(hidden)]
    fn is_canonical(bits: Self::Bits) -> bool;

    /// Returns the sign bit of an encoding.
    #[doc(hidden)]
    fn is_sign_negative(bits: Self::Bits) -> bool;

    /// Rounds an exact value to the format.
    #[doc(hidden)]
    fn round<L: Limbs>(value: &Unrounded<L>, env: &Env) -> (Self::Bits, Flags);

    /// Converts a decoded value of another format, whose NaN payloads have
    /// `payload_bits` bits.
    #[doc(hidden)]
    fn convert_from<L: Limbs>(
        value: Unpacked<L>,
        payload_bits: u32,
        env: &Env,
    ) -> (Self::Bits, Flags);

    /// Adds, or subtracts when `subtract` is set.
    #[doc(hidden)]
    fn add(left: Self::Bits, right: Self::Bits, subtract: bool, env: &Env) -> (Self::Bits, Flags);

    /// Multiplies.
    #[doc(hidden)]
    fn mul(left: Self::Bits, right: Self::Bits, env: &Env) -> (Self::Bits, Flags);

    /// Divides.
    #[doc(hidden)]
    fn div(left: Self::Bits, right: Self::Bits, env: &Env) -> (Self::Bits, Flags);

    /// Returns the square root.
    #[doc(hidden)]
    fn sqrt(value: Self::Bits, env: &Env) -> (Self::Bits, Flags);

    /// Returns `left * right + addend`, rounded once.
    #[doc(hidden)]
    fn mul_add(
        left: Self::Bits,
        right: Self::Bits,
        addend: Self::Bits,
        env: &Env,
    ) -> (Self::Bits, Flags);
}

/// An integer type that stores the encoding of a format.
///
/// The storage types are `u8`, `u16`, `u32`, `u64`, `u128`, and `[u64; N]`
/// for `N` from 3 to 8. An array stores its least significant 64 bits in
/// element 0. The trait is sealed.
pub trait Bits: Sealed + LimbConversion + Copy + Eq + Hash + Debug {}

pub(crate) mod internal {
    //! Conversions that only the engine uses.

    use crate::limbs::Widen;

    /// Converts a storage type to and from the limbs that the engine uses.
    pub trait LimbConversion: Sized {
        /// The limb array that the engine computes on.
        type Limbs: Widen;

        /// Converts the storage value to limbs.
        fn to_limbs(self) -> Self::Limbs;

        /// Converts limbs to the storage value. The value must fit the storage.
        fn from_limbs(limbs: Self::Limbs) -> Self;
    }
}

/// Implements [`Bits`] for an unsigned primitive of at most 64 bits.
macro_rules! small_bits {
    ($($bits:ty),*) => {
        $(
            impl Sealed for $bits {}
            impl Bits for $bits {}
            impl LimbConversion for $bits {
                type Limbs = [u64; 1];

                #[inline]
                fn to_limbs(self) -> [u64; 1] {
                    [u64::from(self)]
                }

                #[inline]
                fn from_limbs([limb]: [u64; 1]) -> Self {
                    Self::try_from(limb).expect("the value fits the storage type")
                }
            }
        )*
    };
}

small_bits!(u8, u16, u32, u64);

impl Sealed for u128 {}
impl Bits for u128 {}
impl LimbConversion for u128 {
    type Limbs = [u64; 2];

    #[inline]
    fn to_limbs(self) -> [u64; 2] {
        let low = u64::try_from(self & Self::from(u64::MAX)).expect("the mask keeps 64 bits");
        let high = u64::try_from(self >> 64).expect("the shift keeps 64 bits");
        [low, high]
    }

    #[inline]
    fn from_limbs([low, high]: [u64; 2]) -> Self {
        Self::from(low) | (Self::from(high) << 64)
    }
}

/// Implements [`Bits`] for the limb arrays that the storage table uses.
macro_rules! array_bits {
    ($($limbs:literal),*) => {
        $(
            impl Sealed for [u64; $limbs] {}
            impl Bits for [u64; $limbs] {}
            impl LimbConversion for [u64; $limbs] {
                type Limbs = Self;

                fn to_limbs(self) -> Self {
                    self
                }

                fn from_limbs(limbs: Self) -> Self {
                    limbs
                }
            }
        )*
    };
}

array_bits!(3, 4, 5, 6, 7, 8);

/// The width `W` in bits, as a key of the storage table.
///
/// `Width<W>` implements [`Storage`] for every `W` from 1 to 512.
pub struct Width<const W: usize>;

/// The storage type of an encoding `W` bits wide.
///
/// The storage is the smallest of `u8`, `u16`, `u32`, `u64`, and `u128`
/// that holds `W` bits, and `[u64; N]` above 128 bits. The trait is sealed.
pub trait Storage: Sealed {
    /// The storage type.
    type Bits: Bits;

    /// The width in bits.
    const WIDTH: u32;
}

macro_rules! storage {
    ($bits:ty => $($width:literal)*) => {
        $(
            impl Sealed for Width<$width> {}
            impl Storage for Width<$width> {
                type Bits = $bits;
                const WIDTH: u32 = $width;
            }
        )*
    };
}

storage!(u8 =>
    1 2 3 4 5 6 7 8
);
storage!(u16 =>
    9 10 11 12 13 14 15 16
);
storage!(u32 =>
    17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32
);
storage!(u64 =>
    33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48
    49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64
);
storage!(u128 =>
    65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80
    81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96
    97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112
    113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128
);
storage!([u64; 3] =>
    129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144
    145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160
    161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176
    177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192
);
storage!([u64; 4] =>
    193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208
    209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 224
    225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240
    241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256
);
storage!([u64; 5] =>
    257 258 259 260 261 262 263 264 265 266 267 268 269 270 271 272
    273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288
    289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304
    305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320
);
storage!([u64; 6] =>
    321 322 323 324 325 326 327 328 329 330 331 332 333 334 335 336
    337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352
    353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368
    369 370 371 372 373 374 375 376 377 378 379 380 381 382 383 384
);
storage!([u64; 7] =>
    385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400
    401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416
    417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 432
    433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448
);
storage!([u64; 8] =>
    449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464
    465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480
    481 482 483 484 485 486 487 488 489 490 491 492 493 494 495 496
    497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512
);

/// The binary floating-point family: a sign bit, `E` exponent bits, and the
/// other bits of the width as the fraction.
///
/// `Enc` sets the special values, the exponent bias, and whether the integer
/// bit is explicit. A format has 2 to 28 exponent bits and at least one
/// fraction bit. The [`X87`] encoding exists only as `Binary<15, X87>` at
/// width 80.
///
/// An unsupported width does not compile:
///
/// ```compile_fail,E0277
/// let _ = floaty::Float::<floaty::Binary<8>, 600>::from_bits([0; 10]);
/// ```
///
/// An invalid layout fails when the compiler generates code for it. `cargo
/// check` does not report it. An exponent too wide for its width fails:
///
/// ```compile_fail,E0080
/// let _ = floaty::Float::<floaty::Binary<9>, 10>::from_bits(0);
/// ```
///
/// The `X87` encoding at another width fails:
///
/// ```compile_fail,E0080
/// let _ = floaty::Float::<floaty::Binary<15, floaty::X87>, 96>::PRECISION;
/// ```
pub struct Binary<const E: u32, Enc: Encoding = Ieee> {
    encoding: PhantomData<Enc>,
}

impl<const E: u32, Enc: Encoding> Sealed for Binary<E, Enc> {}

/// The special values, bias, and integer bit of a binary format.
///
/// The trait is sealed. The encodings are [`Ieee`], [`NoInf`], [`Fnuz`], and
/// [`X87`].
pub trait Encoding: Sealed + 'static {
    /// The encoding rules that the binary engine applies.
    #[doc(hidden)]
    const KIND: EncodingKind;
}

/// The IEEE 754 encoding.
///
/// An exponent field of all ones holds an infinity when the fraction is zero,
/// and a NaN otherwise. The most significant fraction bit is set in a quiet
/// NaN. The bias is `2^(E-1) - 1`. The IEEE 754 binary formats, bfloat16,
/// TF32, and OCP FP8 E5M2 use this encoding.
pub enum Ieee {}

impl Sealed for Ieee {}
impl Encoding for Ieee {
    const KIND: EncodingKind = EncodingKind::Ieee;
}

/// An encoding without infinities, for OCP FP8 E4M3.
///
/// An exponent field and a fraction field of all ones hold a NaN, with either
/// sign. Every other encoding is a number. The bias is `2^(E-1) - 1`. The
/// format has no signaling NaN, so its NaN is quiet.
pub enum NoInf {}

impl Sealed for NoInf {}
impl Encoding for NoInf {
    const KIND: EncodingKind = EncodingKind::NoInf;
}

/// An encoding without infinities and without negative zero, for the FNUZ
/// FP8 formats.
///
/// The sign bit alone holds the one NaN. Every other encoding is a number.
/// The bias is `2^(E-1)`, one larger than in [`Ieee`]. The format has no
/// signaling NaN, so its NaN is quiet.
pub enum Fnuz {}

impl Sealed for Fnuz {}
impl Encoding for Fnuz {
    const KIND: EncodingKind = EncodingKind::Fnuz;
}

/// The x87 extended-precision encoding, with an explicit integer bit.
///
/// The encoding follows the 387 and later processors, as the Intel SDM Volume
/// 1 (253665-093US), section 8.2.2 and Table 8-3 on page 8-14, states. An
/// unnormal, a pseudo-NaN, or a pseudo-infinity is
/// [`Unsupported`](crate::Class::Unsupported). A pseudo-denormal has the
/// value of the normal encoding with exponent field 1. It is subnormal and
/// not canonical.
pub enum X87 {}

impl Sealed for X87 {}
impl Encoding for X87 {
    const KIND: EncodingKind = EncodingKind::X87;
}
