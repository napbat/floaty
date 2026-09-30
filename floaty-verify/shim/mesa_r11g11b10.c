/* Exposes the R11G11B10 conversions of Mesa, which its header defines as
 * static inline functions, to floaty-verify. Each function takes and returns
 * encodings, so no floating-point value crosses the call. */

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "format_r11g11b10f.h"

static float from_bits(uint32_t bits)
{
   float value;
   memcpy(&value, &bits, sizeof value);
   return value;
}

static uint32_t to_bits(float value)
{
   uint32_t bits;
   memcpy(&bits, &value, sizeof bits);
   return bits;
}

uint32_t floaty_mesa_f32_to_uf11(uint32_t bits)
{
   return f32_to_uf11(from_bits(bits));
}

uint32_t floaty_mesa_f32_to_uf10(uint32_t bits)
{
   return f32_to_uf10(from_bits(bits));
}

uint32_t floaty_mesa_uf11_to_f32(uint16_t channel)
{
   return to_bits(uf11_to_f32(channel));
}

uint32_t floaty_mesa_uf10_to_f32(uint16_t channel)
{
   return to_bits(uf10_to_f32(channel));
}
