/*
 * Passes x87 extended values to and from the Intel Decimal Floating-Point
 * Math Library.
 *
 * The library takes and returns the C `long double` by value. Rust has no
 * type for it, so these functions move the value through a 16-byte buffer.
 * The low 10 bytes of the buffer hold the x87 encoding, least significant
 * byte first, and the other bytes are zero.
 *
 * An x87 load or store of the 80-bit format copies the encoding exactly,
 * signaling NaNs and unsupported encodings included.
 */

#include <string.h>

#include "bid_conf.h"
#include "bid_functions.h"

/* The size of the x87 encoding in bytes. */
#define ENCODING_BYTES 10

/* The size of the buffer that holds an encoding. */
#define BUFFER_BYTES 16

static BINARY80 load(const unsigned char *buffer) {
  BINARY80 value = 0;
  memcpy(&value, buffer, ENCODING_BYTES);
  return value;
}

static void store(unsigned char *buffer, BINARY80 value) {
  memset(buffer, 0, BUFFER_BYTES);
  memcpy(buffer, &value, ENCODING_BYTES);
}

BID_UINT32 floaty_binary80_to_bid32(const unsigned char *x, _IDEC_round rounding,
                                    _IDEC_flags *flags) {
  return binary80_to_bid32(load(x), rounding, flags);
}

BID_UINT64 floaty_binary80_to_bid64(const unsigned char *x, _IDEC_round rounding,
                                    _IDEC_flags *flags) {
  return binary80_to_bid64(load(x), rounding, flags);
}

BID_UINT128 floaty_binary80_to_bid128(const unsigned char *x, _IDEC_round rounding,
                                      _IDEC_flags *flags) {
  return binary80_to_bid128(load(x), rounding, flags);
}

void floaty_bid32_to_binary80(unsigned char *result, BID_UINT32 x, _IDEC_round rounding,
                              _IDEC_flags *flags) {
  store(result, bid32_to_binary80(x, rounding, flags));
}

void floaty_bid64_to_binary80(unsigned char *result, BID_UINT64 x, _IDEC_round rounding,
                              _IDEC_flags *flags) {
  store(result, bid64_to_binary80(x, rounding, flags));
}

void floaty_bid128_to_binary80(unsigned char *result, BID_UINT128 x, _IDEC_round rounding,
                               _IDEC_flags *flags) {
  store(result, bid128_to_binary80(x, rounding, flags));
}
