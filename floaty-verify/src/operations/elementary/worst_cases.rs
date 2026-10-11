//! Published hard cases of the elementary functions, which the oracle tests
//! and the margin tests share.

/// The worst cases of `exp` for binary64 of Lefèvre and Muller, "Some Worst
/// Cases for The Table Maker's Dilemma",
/// <https://perso.ens-lyon.fr/jean-michel.muller/TMD.html>, one for each
/// interval of the table.
pub const EXP: [u64; 33] = [
    0xC02E_8BDB_FCD9_144E,
    0xC017_1E0B_869B_5E79,
    0xC000_2393_D597_6769,
    0xBFF2_A9CA_D999_8262,
    0xBFFC_C37E_F7DE_7501,
    0xBFE2_2E24_FA3D_5CF9,
    0xBFED_C2B5_DF1F_7D3D,
    0xBFC2_90EA_09E3_6479,
    0xBF2A_2FEF_EFD5_80DF,
    0xBE4E_D318_EFB6_27EA,
    0xBE04_BD46_601A_E1EF,
    0xBD51_0000_0000_0242,
    0xBD52_0000_0000_0288,
    0xBCF8_0000_0000_0012,
    0xBCC0_0000_0000_0001,
    0x3CAF_FFFF_FFFF_FFFF,
    0x3CFF_FFFF_FFFF_FFE0,
    0x3DF7_FFE7_FFEE_0024,
    0x3DF8_0017_FFED_FFDC,
    0x3E09_E9CB_BFD6_080B,
    0x3E5D_7A7D_8936_09E5,
    0x3F1B_A07D_7325_0DE7,
    0x3F4D_77FD_13D2_7FFF,
    0x3F76_A4D1_AF9C_C989,
    0x3FEA_CCFB_E46B_4EF0,
    0x3FFA_CA7A_E8DA_5A7B,
    0x3FFD_6336_A880_77AA,
    0x4004_2EE3_C7DC_4946,
    0x4018_3D4B_CDEB_B3F4,
    0x4027_6E7E_5D7B_6EAC,
    0x402A_8EAD_058B_C6B8,
    0x4031_D5C2_DAEB_E367,
    0x403C_44CE_0D71_6A1A,
];

/// The worst cases of `log` for binary64 from the same table.
pub const LOG: [u64; 12] = [
    0x3FF0_0000_0000_0001,
    0x4000_0097_6581_CE4E,
    0x4019_1A8D_FF54_0FF7,
    0x401D_E37F_B31F_D5FC,
    0x4022_B119_9E49_7739,
    0x4025_5F0E_AA1B_2FC8,
    0x4028_EDE4_92D9_6072,
    0x4031_1867_637C_BD03,
    0x4033_D9D7_D597_A9DD,
    0x4037_F382_5778_AAAF,
    0x407A_C50B_409C_8AEE,
    0x40FD_E7CD_6751_029A,
];

/// The binary64 argument closest to a multiple of `pi / 2` relative to its
/// size, `6381956970095103 * 2^797`, `4.687e-19` from the multiple, as
/// J.-M. Muller lists it in "Elementary Functions: Algorithms and
/// Implementation", 3rd edition, 2016, chapter 11. Its reduction keeps the
/// fewest digits of any binary64 argument.
pub const REDUCTION: u64 = 0x7506_AC5B_262C_A1FF;

/// Bad cases of `exp` for decimal64, as coefficient and exponent, from
/// Lefèvre, Stehlé, and Zimmermann, "Worst Cases for the Exponential
/// Function in the IEEE 754r decimal64 Format", JNAO 2006, slides 6 and 13
/// to 16, <https://www.vinc17.net/research/slides/jnao2006-05.pdf>.
pub const DECIMAL64_EXP: [(u64, i64); 39] = [
    (6_581_539_478_341_669, -24),
    (2_662_858_264_545_929, -23),
    (3_639_588_333_766_983, -23),
    (6_036_998_017_773_271, -23),
    (6_638_670_361_402_304, -22),
    (9_366_572_213_364_879, -22),
    (7_970_613_003_079_781, -21),
    (3_089_765_552_852_523, -20),
    (1_302_531_956_641_873, -19),
    (2_241_856_702_421_245, -19),
    (7_230_293_679_121_590, -19),
    (5_259_640_428_979_129, -18),
    (9_407_822_313_572_878, -17),
    (1_267_914_924_960_933, -16),
    (5_091_077_534_282_133, -16),
    (3_359_104_074_009_002, -15),
    (1_910_511_686_234_796, -14),
    (2_949_551_257_293_143, -13),
    (5_879_131_381_356_093, -13),
    (7_906_867_968_553_504, -16),
    (1_548_443_067_391_468, -18),
    (2_953_379_504_777_270, -16),
    (3_897_940_992_403_028, -24),
    (4_230_932_991_049_603, -24),
    (4_291_382_990_792_016, -24),
    (4_581_289_989_505_891, -24),
    (5_999_879_998_200_072, -25),
    (6_000_119_998_199_928, -25),
    (1_019_999_999_994_798, -26),
    (1_039_999_999_994_592, -26),
    (1_099_999_999_999_395, -27),
    (1_199_999_999_999_280, -27),
    (1_199_999_999_999_928, -28),
    (1_399_999_999_999_902, -28),
    (1_999_999_999_999_980, -29),
    (2_999_999_999_999_955, -29),
    (1_999_999_999_999_998, -30),
    (3_999_999_999_999_992, -30),
    (9_999_999_999_999_995, -31),
];
