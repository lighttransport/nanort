/*
Copyright (c) 2013, Syoyo Fujita.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#include "eson.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>

#ifdef _WIN32
#include <Windows.h>  // File mapping
//#error TODO
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

using namespace eson;

//
// --
//

uint8_t* Value::Serialize(uint8_t* p) const {
  // printf("ty = %d\n", type_);

  uint8_t* ptr = p;
  switch (type_) {
    case FLOAT64_TYPE:
      memcpy(ptr, &float64_, sizeof(double));
      ptr += sizeof(double);
      break;
    case INT64_TYPE: {
      (*(reinterpret_cast<int64_t*>(ptr))) = int64_;
      ptr += sizeof(int64_t);
    } break;
    case STRING_TYPE: {
      // len(64bit) + string
      (*(reinterpret_cast<int64_t*>(ptr))) = size_;
      // printf("//str data len = %d\n", (int)size_);
      ptr += sizeof(int64_t);
      memcpy(ptr, string_->c_str(), size_);
      // printf("//str val = %s\n", string_->c_str());
      ptr += size_;
    } break;
    case BINARY_TYPE: {
      // len(64bit) + bindata
      (*(reinterpret_cast<int64_t*>(ptr))) = size_;
      // printf("//bin data len = %d\n", (int)size_);
      ptr += sizeof(int64_t);
      memcpy(ptr, binary_->ptr, size_);
      ptr += size_;
    } break;
    case OBJECT_TYPE: {
      // First compute object size
      int64_t object_size = Size();
      // printf("obj size = %lld\n", object_size);

      (*(reinterpret_cast<int64_t*>(ptr))) = size_;
      ptr += sizeof(int64_t);

      // Serialize key-value pairs.
      for (Object::const_iterator it = object_->begin(); it != object_->end();
           ++it) {
        // Emit type tag.
        char ty = static_cast<char>(it->second.type_);
        (*(reinterpret_cast<char*>(ptr))) = ty;
        ptr++;

        // Emit key
        const std::string& key = it->first;
        memcpy(ptr, key.c_str(), key.size());
        // printf("[[key len = %d\n", (int)key.size());
        ptr += key.size();
        (*(reinterpret_cast<char*>(ptr))) = '\0';  // null terminate
        ptr++;

        // Emit element
        ptr = it->second.Serialize(ptr);
      }
    } break;
    default:
      assert(0);
      break;
  }

  return ptr;
}

// Forward decl.
static const uint8_t* ParseElement(std::stringstream& err, Object& o,
                                   const uint8_t* p);

static const uint8_t* ReadKey(std::string& key, const uint8_t* p) {
  key = std::string(reinterpret_cast<const char*>(p));
  // printf("key = [ %s ]\n", key.c_str());

  // printf("key len = %d\n", (int)key.length());
  // printf("str len = %d\n", (int)strlen((char*)p));

  p += key.length() + 1;  // + '\0'

  return p;
}

static const uint8_t* ReadFloat64(double& v, const uint8_t* p) {
  v = *(reinterpret_cast<const double*>(p));
  p += sizeof(double);

  // printf("fp64 = %f ]\n", v);

  return p;
}

static const uint8_t* ReadInt64(int64_t& v, const uint8_t* p) {
  v = *(reinterpret_cast<const int64_t*>(p));
  p += sizeof(int64_t);

  // printf("int64 = %lld ]\n", v);

  return p;
}

static const uint8_t* ReadString(std::string& s, const uint8_t* p) {
  // N + string data.
  int64_t n = *(reinterpret_cast<const int64_t*>(p));
  p += sizeof(int64_t);

  assert(n >= 0);

  s = std::string(reinterpret_cast<const char*>(p), n);
  p += n;

  // printf("str = %s\n", s.c_str());

  return p;
}

static const uint8_t* ReadBinary(const uint8_t*& b, int64_t& n,
                                 const uint8_t* p) {
  // N + bin data.
  n = *(reinterpret_cast<const int64_t*>(p));
  p += sizeof(int64_t);

  assert(n >= 0);

  // Just returns pointer address.
  b = p;
  p += n;

  // printf("bin len = %d\n", (int)n);

  return p;
}

static const uint8_t* ReadObject(Object& o, const uint8_t* p) {
  // N + object data.
  int64_t n = *(reinterpret_cast<const int64_t*>(p));
  p += sizeof(int64_t);

  assert(n >= 0);

  std::stringstream ss;
  const uint8_t* ptr = ParseElement(ss, o, p);

  return ptr;
}

static const uint8_t* ParseElement(std::stringstream& err, Object& o,
                                   const uint8_t* p) {
  const uint8_t* ptr = p;

  // Read tag;
  Type type = (Type) * (reinterpret_cast<const char*>(ptr));
  ptr++;
  // printf("[Parse] ty = %d\n", type);

  std::string key;
  ptr = ReadKey(key, ptr);

  switch (type) {
    case FLOAT64_TYPE: {
      double val;
      ptr = ReadFloat64(val, ptr);
      o[key] = Value(val);
    } break;
    case INT64_TYPE: {
      int64_t val;
      ptr = ReadInt64(val, ptr);
      o[key] = Value(val);
    } break;
    case STRING_TYPE: {
      std::string str;
      ptr = ReadString(str, ptr);
      o[key] = Value(str);
    } break;
    case BINARY_TYPE: {
      const uint8_t* bin_ptr;
      int64_t bin_size;
      ptr = ReadBinary(bin_ptr, bin_size, ptr);
      // for (int i = 0; i < bin_size; i++) {
      //  printf("b[%d] = %d\n", i, bin_ptr[i]);
      //}
      o[key] = Value(bin_ptr, bin_size);
    } break;
    case OBJECT_TYPE: {
      int64_t read_size = 0;
      Object obj;
      ptr = ReadObject(obj, ptr);

      o[key] = Value(obj);
    } break;
    default:
      assert(0);
      break;
  }

  return ptr;
}

std::string eson::Parse(Value& v, const uint8_t* p) {
  std::stringstream err;

  const uint8_t* ptr = p;

  //
  // == toplevel element
  //

  // Read total size.
  int64_t sz = *(reinterpret_cast<const int64_t*>(ptr));
  ptr += sizeof(int64_t);
  // printf("[Parse] total size = %lld\n", sz);
  assert(sz > 0);

  Object obj;

  int64_t read_size = 0;
  while (read_size < sz) {
    ptr = ParseElement(err, obj, ptr);
    read_size = ptr - p;
    // printf("read_size = %d, total = %d\n", (int)read_size, (int)sz);
  }

  v = Value(obj);

  return err.str();
}
