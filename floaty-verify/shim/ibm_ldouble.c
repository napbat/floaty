/*
 * Runs the IBM long double arithmetic of libgcc on PowerPC, the IBM long
 * double functions of glibc's libm, and the PowerPC fused multiply-add
 * instructions, as an oracle.
 *
 * The program is built for powerpc64le and runs under qemu-ppc64le. It reads
 * one case per line from standard input:
 *
 *     <operation> <rounding> <a_hi> <a_lo> <b_hi> <b_lo>
 *     <function> <rounding> <a_hi> <a_lo> [<b_hi> <b_lo> [<c_hi> <c_lo>]]
 *     <instruction> <rounding> <a> <c> <b>
 *
 * The operation is `add`, `sub`, `mul`, or `div`. The function is a libm
 * function on one, two, or three IBM long double operands: `sqrtl`,
 * `nextupl`, `nextdownl`, `iscanonicall`, `logbl`, `copysignl`, `fmodl`,
 * `remainderl`, `fmal`, a rounding to an integral value (`floorl`, `ceill`,
 * `truncl`, `roundl`, `roundevenl`, `rintl`, or `nearbyintl`), `scalbnl`,
 * `ilogbl`, a minimum or maximum operation (`fmaxl`, `fminl`, `fmaximuml`,
 * `fminimuml`, `fmaximum_numl`, `fminimum_numl`, `fmaximum_magl`,
 * `fminimum_magl`, `fmaximum_mag_numl`, or `fminimum_mag_numl`),
 * `totalorderl`, `totalordermagl`, `llrintl`, `lroundl`, `getpayloadl`,
 * `setpayloadl`, or `setpayloadsigl`.
 * `iscanonicall` gives its integer result as the value of the high half, and
 * a zero low half. `ilogbl`, `totalorderl`, `totalordermagl`, `llrintl`, and
 * `lroundl` give the bits of their integer result, sign-extended to 64 bits,
 * as the high half, and a zero low half. `scalbnl` reads its exponent from
 * the bits of the high half of its second operand, a 64-bit two's complement
 * integer in the range of `int`. The instruction is `fmadd` or `fmsub`,
 * which compute `a * c + b` and `a * c - b` with the operands in the order
 * FRA, FRC, FRB. The rounding direction is `nearest`, `zero`, `up`, or
 * `down`. Each operand is a binary64 bit pattern in exactly 16 hexadecimal
 * digits. For each case the program writes one line:
 *
 *     <result_hi> <result_lo> <flags>
 *     <result> <flags>
 *
 * The results have 16 hexadecimal digits. The flags have 2 hexadecimal
 * digits: invalid 0x01, divide by zero 0x02, overflow 0x04, underflow 0x08,
 * and inexact 0x10. A malformed line stops the program with status 2.
 *
 * The compiler cannot fold or move the libgcc and libm calls. The operands
 * come from standard input, and each call goes through a pointer to an
 * external symbol, so the compiler treats the callee as an unknown function.
 * The instructions are volatile inline assembly. The build uses
 * `-frounding-math`.
 */

#include <fenv.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * The libgcc routines. The default `long double` of this compiler is IEEE
 * binary128, so the declarations name the IBM type explicitly. The routines
 * return the high half in f1 and the low half in f2.
 */
__ibm128 __gcc_qadd(double a, double aa, double c, double cc);
__ibm128 __gcc_qsub(double a, double aa, double c, double cc);
__ibm128 __gcc_qmul(double a, double aa, double c, double cc);
__ibm128 __gcc_qdiv(double a, double aa, double c, double cc);

typedef __ibm128 (*operation_function)(double, double, double, double);

/*
 * The IBM long double functions of glibc. The default `long double` of this
 * compiler is binary128, whose functions have other names, so each assembler
 * label names the IBM entry point of the static libm.
 */
__ibm128 glibc_sqrtl(__ibm128 x) __asm__("__sqrtl");
__ibm128 glibc_nextupl(__ibm128 x) __asm__("__nextupl");
__ibm128 glibc_nextdownl(__ibm128 x) __asm__("__nextdownl");
int glibc_iscanonicall(__ibm128 x) __asm__("__iscanonicall");
__ibm128 glibc_logbl(__ibm128 x) __asm__("__logbl");
__ibm128 glibc_copysignl(__ibm128 x, __ibm128 y) __asm__("__copysignl");
__ibm128 glibc_fmodl(__ibm128 x, __ibm128 y) __asm__("__fmodl");
__ibm128 glibc_remainderl(__ibm128 x, __ibm128 y) __asm__("__remainderl");
__ibm128 glibc_fmal(__ibm128 x, __ibm128 y, __ibm128 z) __asm__("__fmal");
__ibm128 glibc_floorl(__ibm128 x) __asm__("__floorl");
__ibm128 glibc_ceill(__ibm128 x) __asm__("__ceill");
__ibm128 glibc_truncl(__ibm128 x) __asm__("__truncl");
__ibm128 glibc_roundl(__ibm128 x) __asm__("__roundl");
__ibm128 glibc_roundevenl(__ibm128 x) __asm__("__roundevenl");
__ibm128 glibc_rintl(__ibm128 x) __asm__("__rintl");
__ibm128 glibc_nearbyintl(__ibm128 x) __asm__("__nearbyintl");
__ibm128 glibc_scalbnl(__ibm128 x, int n) __asm__("__scalbnl");
int glibc_ilogbl(__ibm128 x) __asm__("__ilogbl");
__ibm128 glibc_fmaxl(__ibm128 x, __ibm128 y) __asm__("__fmaxl");
__ibm128 glibc_fminl(__ibm128 x, __ibm128 y) __asm__("__fminl");
__ibm128 glibc_fmaximuml(__ibm128 x, __ibm128 y) __asm__("__fmaximuml");
__ibm128 glibc_fminimuml(__ibm128 x, __ibm128 y) __asm__("__fminimuml");
__ibm128 glibc_fmaximum_numl(__ibm128 x, __ibm128 y) __asm__("__fmaximum_numl");
__ibm128 glibc_fminimum_numl(__ibm128 x, __ibm128 y) __asm__("__fminimum_numl");
__ibm128 glibc_fmaximum_magl(__ibm128 x, __ibm128 y) __asm__("__fmaximum_magl");
__ibm128 glibc_fminimum_magl(__ibm128 x, __ibm128 y) __asm__("__fminimum_magl");
__ibm128 glibc_fmaximum_mag_numl(__ibm128 x, __ibm128 y) __asm__("__fmaximum_mag_numl");
__ibm128 glibc_fminimum_mag_numl(__ibm128 x, __ibm128 y) __asm__("__fminimum_mag_numl");
int glibc_totalorderl(const __ibm128 *x, const __ibm128 *y) __asm__("__totalorderl");
int glibc_totalordermagl(const __ibm128 *x, const __ibm128 *y) __asm__("__totalordermagl");
long long glibc_llrintl(__ibm128 x) __asm__("__llrintl");
long glibc_lroundl(__ibm128 x) __asm__("__lroundl");
__ibm128 glibc_getpayloadl(const __ibm128 *x) __asm__("__getpayloadl");
int glibc_setpayloadl(__ibm128 *result, __ibm128 payload) __asm__("__setpayloadl");
int glibc_setpayloadsigl(__ibm128 *result, __ibm128 payload) __asm__("__setpayloadsigl");

/*
 * Returns a long double whose high half holds the bits of `value`, and whose
 * low half is zero. The integer operations give no floating-point flags.
 */
static __ibm128 integer_result(long long value) {
  uint64_t bits = (uint64_t)value;
  double high;
  memcpy(&high, &bits, sizeof high);
  return __builtin_pack_ibm128(high, 0.0);
}

/* Returns the integer that the bits of the high half of `x` hold. */
static int integer_operand(__ibm128 x) {
  double high = __builtin_unpack_ibm128(x, 0);
  int64_t bits;
  memcpy(&bits, &high, sizeof bits);
  return (int)bits;
}

static __ibm128 scale(__ibm128 x, __ibm128 n) { return glibc_scalbnl(x, integer_operand(n)); }
static __ibm128 exponent(__ibm128 x) { return integer_result(glibc_ilogbl(x)); }
static __ibm128 total_order(__ibm128 x, __ibm128 y) {
  return integer_result(glibc_totalorderl(&x, &y));
}
static __ibm128 total_order_magnitude(__ibm128 x, __ibm128 y) {
  return integer_result(glibc_totalordermagl(&x, &y));
}
static __ibm128 rint_integer(__ibm128 x) { return integer_result(glibc_llrintl(x)); }
static __ibm128 round_integer(__ibm128 x) { return integer_result(glibc_lroundl(x)); }

static __ibm128 get_payload(__ibm128 x) { return glibc_getpayloadl(&x); }
static __ibm128 set_payload(__ibm128 x) {
  __ibm128 result;
  glibc_setpayloadl(&result, x);
  return result;
}
static __ibm128 set_payload_signaling(__ibm128 x) {
  __ibm128 result;
  glibc_setpayloadsigl(&result, x);
  return result;
}

/* Returns `iscanonicall` as a long double, for the one-operand table. */
static __ibm128 canonical(__ibm128 x) {
  return __builtin_pack_ibm128((double)glibc_iscanonicall(x), 0.0);
}

typedef __ibm128 (*unary_function)(__ibm128);
typedef __ibm128 (*binary_function)(__ibm128, __ibm128);
typedef __ibm128 (*ternary_function)(__ibm128, __ibm128, __ibm128);

/* A libm function, with its operand count. */
struct library_function {
  const char *name;
  int operand_count;
  unary_function unary;
  binary_function binary;
  ternary_function ternary;
};

static const struct library_function FUNCTIONS[] = {
    {"sqrtl", 1, glibc_sqrtl, NULL, NULL},
    {"nextupl", 1, glibc_nextupl, NULL, NULL},
    {"nextdownl", 1, glibc_nextdownl, NULL, NULL},
    {"iscanonicall", 1, canonical, NULL, NULL},
    {"logbl", 1, glibc_logbl, NULL, NULL},
    {"copysignl", 2, NULL, glibc_copysignl, NULL},
    {"fmodl", 2, NULL, glibc_fmodl, NULL},
    {"remainderl", 2, NULL, glibc_remainderl, NULL},
    {"fmal", 3, NULL, NULL, glibc_fmal},
    {"floorl", 1, glibc_floorl, NULL, NULL},
    {"ceill", 1, glibc_ceill, NULL, NULL},
    {"truncl", 1, glibc_truncl, NULL, NULL},
    {"roundl", 1, glibc_roundl, NULL, NULL},
    {"roundevenl", 1, glibc_roundevenl, NULL, NULL},
    {"rintl", 1, glibc_rintl, NULL, NULL},
    {"nearbyintl", 1, glibc_nearbyintl, NULL, NULL},
    {"scalbnl", 2, NULL, scale, NULL},
    {"ilogbl", 1, exponent, NULL, NULL},
    {"getpayloadl", 1, get_payload, NULL, NULL},
    {"setpayloadl", 1, set_payload, NULL, NULL},
    {"setpayloadsigl", 1, set_payload_signaling, NULL, NULL},
    {"fmaxl", 2, NULL, glibc_fmaxl, NULL},
    {"fminl", 2, NULL, glibc_fminl, NULL},
    {"fmaximuml", 2, NULL, glibc_fmaximuml, NULL},
    {"fminimuml", 2, NULL, glibc_fminimuml, NULL},
    {"fmaximum_numl", 2, NULL, glibc_fmaximum_numl, NULL},
    {"fminimum_numl", 2, NULL, glibc_fminimum_numl, NULL},
    {"fmaximum_magl", 2, NULL, glibc_fmaximum_magl, NULL},
    {"fminimum_magl", 2, NULL, glibc_fminimum_magl, NULL},
    {"fmaximum_mag_numl", 2, NULL, glibc_fmaximum_mag_numl, NULL},
    {"fminimum_mag_numl", 2, NULL, glibc_fminimum_mag_numl, NULL},
    {"totalorderl", 2, NULL, total_order, NULL},
    {"totalordermagl", 2, NULL, total_order_magnitude, NULL},
    {"llrintl", 1, rint_integer, NULL, NULL},
    {"lroundl", 1, round_integer, NULL, NULL},
};

/*
 * Returns `a * c + b` by the `fmadd` instruction. The memory clobber keeps
 * the instruction between the calls that set the rounding direction and read
 * the flags.
 */
static double fused_multiply_add(double a, double c, double b) {
  double result;
  __asm__ volatile("fmadd %0,%1,%2,%3" : "=d"(result) : "d"(a), "d"(c), "d"(b) : "memory");
  return result;
}

/* Returns `a * c - b` by the `fmsub` instruction, as `fused_multiply_add`. */
static double fused_multiply_subtract(double a, double c, double b) {
  double result;
  __asm__ volatile("fmsub %0,%1,%2,%3" : "=d"(result) : "d"(a), "d"(c), "d"(b) : "memory");
  return result;
}

typedef double (*instruction_function)(double, double, double);

/* The number of hexadecimal digits of one binary64 bit pattern. */
#define HEX_DIGITS 16

/* The longest input line that the program accepts, with its newline. */
#define LINE_CAPACITY 128

/* The output flag bits. */
#define FLAG_INVALID 0x01u
#define FLAG_DIVIDE_BY_ZERO 0x02u
#define FLAG_OVERFLOW 0x04u
#define FLAG_UNDERFLOW 0x08u
#define FLAG_INEXACT 0x10u

struct name_value {
  const char *name;
  int value;
};

static const struct name_value ROUNDINGS[] = {
    {"nearest", FE_TONEAREST},
    {"zero", FE_TOWARDZERO},
    {"up", FE_UPWARD},
    {"down", FE_DOWNWARD},
};

static void fail(unsigned long line_number, const char *message) {
  fprintf(stderr, "ibm_ldouble: line %lu: %s\n", line_number, message);
  exit(2);
}

/* Returns the value of a hexadecimal digit, or -1 for another character. */
static int digit_value(char digit) {
  if (digit >= '0' && digit <= '9') {
    return digit - '0';
  }
  if (digit >= 'a' && digit <= 'f') {
    return digit - 'a' + 10;
  }
  if (digit >= 'A' && digit <= 'F') {
    return digit - 'A' + 10;
  }
  return -1;
}

/*
 * Reads one field of exactly 16 hexadecimal digits at `*cursor`, and moves
 * the cursor past it and past one following space or newline.
 */
static int read_bits(const char **cursor, uint64_t *bits) {
  const char *text = *cursor;
  uint64_t value = 0;
  for (int index = 0; index < HEX_DIGITS; index++) {
    int digit = digit_value(text[index]);
    if (digit < 0) {
      return 0;
    }
    value = (value << 4) | (uint64_t)digit;
  }
  char end = text[HEX_DIGITS];
  if (end != ' ' && end != '\n' && end != '\0') {
    return 0;
  }
  *cursor = end == '\0' ? text + HEX_DIGITS : text + HEX_DIGITS + 1;
  *bits = value;
  return 1;
}

/*
 * Reads one word at `*cursor` that ends with a space, and moves the cursor
 * past the space. Returns the length of the word, or 0 when no space ends
 * it.
 */
static size_t read_word(const char **cursor, const char **word) {
  const char *space = strchr(*cursor, ' ');
  if (space == NULL) {
    return 0;
  }
  *word = *cursor;
  size_t length = (size_t)(space - *cursor);
  *cursor = space + 1;
  return length;
}

static int word_is(const char *word, size_t length, const char *name) {
  return strlen(name) == length && memcmp(word, name, length) == 0;
}

static operation_function find_operation(const char *word, size_t length) {
  if (word_is(word, length, "add")) {
    return __gcc_qadd;
  }
  if (word_is(word, length, "sub")) {
    return __gcc_qsub;
  }
  if (word_is(word, length, "mul")) {
    return __gcc_qmul;
  }
  if (word_is(word, length, "div")) {
    return __gcc_qdiv;
  }
  return NULL;
}

static const struct library_function *find_function(const char *word, size_t length) {
  for (size_t index = 0; index < sizeof FUNCTIONS / sizeof FUNCTIONS[0]; index++) {
    if (word_is(word, length, FUNCTIONS[index].name)) {
      return &FUNCTIONS[index];
    }
  }
  return NULL;
}

static instruction_function find_instruction(const char *word, size_t length) {
  if (word_is(word, length, "fmadd")) {
    return fused_multiply_add;
  }
  if (word_is(word, length, "fmsub")) {
    return fused_multiply_subtract;
  }
  return NULL;
}

static int find_rounding(const char *word, size_t length, int *mode) {
  for (size_t index = 0; index < sizeof ROUNDINGS / sizeof ROUNDINGS[0]; index++) {
    if (word_is(word, length, ROUNDINGS[index].name)) {
      *mode = ROUNDINGS[index].value;
      return 1;
    }
  }
  return 0;
}

static double from_bits(uint64_t bits) {
  double value;
  memcpy(&value, &bits, sizeof value);
  return value;
}

static uint64_t to_bits(double value) {
  uint64_t bits;
  memcpy(&bits, &value, sizeof bits);
  return bits;
}

/* Returns the output flag bits of the raised `FE_*` exceptions. */
static unsigned output_flags(int raised) {
  unsigned flags = 0;
  if (raised & FE_INVALID) {
    flags |= FLAG_INVALID;
  }
  if (raised & FE_DIVBYZERO) {
    flags |= FLAG_DIVIDE_BY_ZERO;
  }
  if (raised & FE_OVERFLOW) {
    flags |= FLAG_OVERFLOW;
  }
  if (raised & FE_UNDERFLOW) {
    flags |= FLAG_UNDERFLOW;
  }
  if (raised & FE_INEXACT) {
    flags |= FLAG_INEXACT;
  }
  return flags;
}

/* Writes the low `count` hexadecimal digits of `value`, in lowercase. */
static void put_hex(char *text, uint64_t value, int count) {
  static const char DIGITS[] = "0123456789abcdef";
  for (int index = count - 1; index >= 0; index--) {
    text[index] = DIGITS[value & 0xF];
    value >>= 4;
  }
}

/*
 * Writes one output line: each result in 16 hexadecimal digits, and the flags
 * in 2.
 */
static void write_outcome(const uint64_t *results, int count, int raised) {
  /* Two results, two flag digits, two spaces, and a newline. */
  char output[2 * HEX_DIGITS + 2 + 3];
  char *text = output;
  for (int index = 0; index < count; index++) {
    put_hex(text, results[index], HEX_DIGITS);
    text += HEX_DIGITS;
    *text++ = ' ';
  }
  put_hex(text, output_flags(raised), 2);
  text += 2;
  *text++ = '\n';
  fwrite(output, 1, (size_t)(text - output), stdout);
}

int main(void) {
  char line[LINE_CAPACITY];
  unsigned long line_number = 0;
  while (fgets(line, sizeof line, stdin) != NULL) {
    line_number++;
    if (strchr(line, '\n') == NULL && !feof(stdin)) {
      fail(line_number, "the line is too long");
    }
    const char *cursor = line;
    const char *word = "";
    size_t length = read_word(&cursor, &word);
    operation_function operation = find_operation(word, length);
    const struct library_function *function = find_function(word, length);
    instruction_function instruction = find_instruction(word, length);
    if (operation == NULL && function == NULL && instruction == NULL) {
      fail(line_number, "unknown operation");
    }
    length = read_word(&cursor, &word);
    int mode = FE_TONEAREST;
    if (!find_rounding(word, length, &mode)) {
      fail(line_number, "unknown rounding direction");
    }
    int operand_count = operation != NULL  ? 4
                        : function != NULL ? 2 * function->operand_count
                                           : 3;
    uint64_t operands[6];
    for (int index = 0; index < operand_count; index++) {
      if (!read_bits(&cursor, &operands[index])) {
        fail(line_number, "an operand is not 16 hexadecimal digits");
      }
    }
    if (*cursor != '\0') {
      fail(line_number, "the line has extra text");
    }

    if (fesetround(mode) != 0) {
      fail(line_number, "fesetround failed");
    }
    uint64_t results[2];
    int result_count;
    int raised;
    feclearexcept(FE_ALL_EXCEPT);
    if (operation != NULL) {
      __ibm128 result = operation(from_bits(operands[0]), from_bits(operands[1]),
                                  from_bits(operands[2]), from_bits(operands[3]));
      raised = fetestexcept(FE_ALL_EXCEPT);
      results[0] = to_bits(__builtin_unpack_ibm128(result, 0));
      results[1] = to_bits(__builtin_unpack_ibm128(result, 1));
      result_count = 2;
    } else if (function != NULL) {
      __ibm128 values[3];
      for (int index = 0; index < function->operand_count; index++) {
        values[index] = __builtin_pack_ibm128(from_bits(operands[2 * index]),
                                              from_bits(operands[2 * index + 1]));
      }
      __ibm128 result = function->operand_count == 1   ? function->unary(values[0])
                        : function->operand_count == 2 ? function->binary(values[0], values[1])
                                                       : function->ternary(values[0], values[1],
                                                                           values[2]);
      raised = fetestexcept(FE_ALL_EXCEPT);
      results[0] = to_bits(__builtin_unpack_ibm128(result, 0));
      results[1] = to_bits(__builtin_unpack_ibm128(result, 1));
      result_count = 2;
    } else {
      double result =
          instruction(from_bits(operands[0]), from_bits(operands[1]), from_bits(operands[2]));
      raised = fetestexcept(FE_ALL_EXCEPT);
      results[0] = to_bits(result);
      result_count = 1;
    }
    fesetround(FE_TONEAREST);
    write_outcome(results, result_count, raised);
  }
  if (ferror(stdin)) {
    fail(line_number, "standard input cannot be read");
  }
  if (fflush(stdout) != 0) {
    fail(line_number, "standard output cannot be written");
  }
  return 0;
}
