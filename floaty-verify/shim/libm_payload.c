/* Exposes the NaN payload functions of the host glibc libm to floaty-verify:
 * getpayload, setpayload, and setpayloadsig of C23 annex F.10.13, for float,
 * double, the x87 long double, and _Float128. Each function reads an
 * encoding from 16 bytes and writes the encoding of the result to 16 bytes,
 * in the byte order of the host, so no floating-point value crosses the
 * call. */

#define __STDC_WANT_IEC_60559_BFP_EXT__ 1
#define __STDC_WANT_IEC_60559_TYPES_EXT__ 1

#include <math.h>
#include <string.h>

/* The operation codes of floaty-verify's `libm::Payload`. */
enum { GET = 0, SET = 1, SET_SIGNALING = 2 };

/* Defines floaty_libm_<name>, which runs one payload operation on a format
 * of `size` bytes. setpayload and setpayloadsig give +0 for a payload that
 * they do not admit. */
#define PAYLOAD_FUNCTION(name, type, size, get, set, set_signaling)            \
   void floaty_libm_##name(int operation, const unsigned char *input,       \
                           unsigned char *output)                           \
   {                                                                         \
      type value;                                                            \
      type result;                                                           \
      memset(&value, 0, sizeof value);                                       \
      memset(&result, 0, sizeof result);                                     \
      memcpy(&value, input, size);                                           \
      switch (operation) {                                                   \
      case GET:                                                              \
         result = get(&value);                                               \
         break;                                                              \
      case SET:                                                              \
         set(&result, value);                                                \
         break;                                                              \
      default:                                                               \
         set_signaling(&result, value);                                      \
         break;                                                              \
      }                                                                      \
      memset(output, 0, 16);                                                 \
      memcpy(output, &result, size);                                         \
   }

PAYLOAD_FUNCTION(float, float, 4, getpayloadf, setpayloadf, setpayloadsigf)
PAYLOAD_FUNCTION(double, double, 8, getpayload, setpayload, setpayloadsig)
PAYLOAD_FUNCTION(long_double, long double, 10, getpayloadl, setpayloadl, setpayloadsigl)
PAYLOAD_FUNCTION(float128, _Float128, 16, getpayloadf128, setpayloadf128, setpayloadsigf128)
