/*
Copyright (c) 2013, Light Transport Entertainment Inc.
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
#ifndef __ESON_H__
#define __ESON_H__

#include <stdint.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>

#include <map>
#include <string>
#include <vector>

namespace eson {

typedef enum {
  NULL_TYPE = 0,
  FLOAT64_TYPE = 1,
  INT64_TYPE = 2,
  BOOL_TYPE = 3,
  STRING_TYPE = 4,
  ARRAY_TYPE = 5,  // @todo
  BINARY_TYPE = 6,
  OBJECT_TYPE = 7,
} Type;

class Value {
 public:
  typedef struct {
    const uint8_t* ptr;
    int64_t size;
  } Binary;

  typedef std::vector<Value> Array;
  typedef std::map<std::string, Value> Object;

 protected:
  int type_;               // Data type
  mutable uint64_t size_;  // Data size
  mutable bool dirty_;
  union {
    bool boolean_;
    int64_t int64_;
    double float64_;
    std::string* string_;
    Binary* binary_;
    Array* array_;
    Object* object_;
  };

 public:
  Value() : type_(NULL_TYPE), dirty_(true) {}

  explicit Value(bool b) : type_(BOOL_TYPE), dirty_(false) {
    boolean_ = b;
    size_ = 1;
  }
  explicit Value(int64_t i) : type_(INT64_TYPE), dirty_(false) {
    int64_ = i;
    size_ = 8;
  }
  explicit Value(double n) : type_(FLOAT64_TYPE), dirty_(false) {
    float64_ = n;
    size_ = 8;
  }
  explicit Value(const std::string& s) : type_(STRING_TYPE), dirty_(false) {
    string_ = new std::string(s);
    size_ = string_->size();
  }
  explicit Value(const uint8_t* p, uint64_t n)
      : type_(BINARY_TYPE), dirty_(false) {
    binary_ = new Binary;
    binary_->ptr = p;  // Just save a pointer.
    binary_->size = n;
    size_ = n;
  }
  explicit Value(const Array& a) : type_(ARRAY_TYPE), dirty_(true) {
    array_ = new Array(a);
    size_ = ComputeArraySize();
  }
  explicit Value(const Object& o) : type_(OBJECT_TYPE), dirty_(true) {
    object_ = new Object(o);
    size_ = ComputeObjectSize();
  }
  ~Value(){};

  /// Compute size of array element.
  int64_t ComputeArraySize() const {
    assert(type_ == ARRAY_TYPE);

    int64_t array_size = 0;

    assert(array_->size() > 0);

    char base_element_type = (*array_)[0].Type();

    //
    // Elements in the array must be all same type.
    //

    for (size_t i = 0; i < array_->size(); i++) {
      char element_type = (*array_)[i].Type();
      assert(base_element_type == element_type);
      // @todo
      assert(0);
    }

    return -1;  // @todo
  }

  /// Compute object size.
  int64_t ComputeObjectSize() const {
    assert(type_ == OBJECT_TYPE);

    int64_t object_size = 0;

    for (Object::const_iterator it = object_->begin(); it != object_->end();
         ++it) {
      const std::string& key = it->first;
      int64_t key_len = key.length() + 1;  // + '\0'
      int64_t data_len = it->second.ComputeSize();
      // printf("key len = %lld\n", key_len);
      // printf("data len = %lld\n", data_len);
      object_size += key_len + data_len + 1;  // +1 = tag size.
    }

    return object_size;
  }

  /// Compute data size.
  int64_t ComputeSize() const {
    switch (type_) {
      case INT64_TYPE:
        return 8;
        break;
      case FLOAT64_TYPE:
        return 8;
        break;
      case STRING_TYPE:
        return string_->size() + sizeof(int64_t);  // N + str data
        break;
      case BINARY_TYPE:
        return size_ + sizeof(int64_t);  // N + bin data
        break;
      case ARRAY_TYPE:
        return ComputeArraySize() + sizeof(int64_t);  // datalen + N
        break;
      case OBJECT_TYPE:
        return ComputeObjectSize() + sizeof(int64_t);  // datalen + N
        break;
      default:
        assert(0);
        break;
    }
    assert(0);
    return -1;  // Never come here.
  }

  int64_t Size() const {
    if (!dirty_) {
      return size_;
    } else {
      // Recompute data size.
      size_ = ComputeSize();
      dirty_ = false;
      return size_;
    }
  }

  const char Type() const { return (const char)type_; }

  const bool IsInt64() const { return (type_ == FLOAT64_TYPE); }

  const bool IsFloat64() const { return (type_ == FLOAT64_TYPE); }

  const bool IsString() const { return (type_ == STRING_TYPE); }

  const bool IsBinary() const { return (type_ == BINARY_TYPE); }

  const bool IsArray() const { return (type_ == ARRAY_TYPE); }

  const bool IsObject() const { return (type_ == OBJECT_TYPE); }

  // Accessor
  template <typename T>
  const T& Get() const;
  template <typename T>
  T& Get();

  // Lookup value from an array
  const Value& Get(int64_t idx) const {
    static Value null_value;
    assert(IsArray());
    assert(idx >= 0);
    return ((uint64_t)idx < array_->size()) ? (*array_)[idx] : null_value;
  }

  // Lookup value from a key-value pair
  const Value& Get(const std::string& key) const {
    static Value null_value;
    assert(IsObject());
    Object::const_iterator it = object_->find(key);
    return (it != object_->end()) ? it->second : null_value;
  }

  // Valid only for object type.
  bool Has(const std::string& key) const {
    if (!IsObject()) return false;
    Object::const_iterator it = object_->find(key);
    return (it != object_->end()) ? true : false;
  }

  // Serialize data to memory 'p'.
  // Memory of 'p' must be allocated by app before calling this function.
  // (size can be obtained by calling 'Size' function.
  // Return next data location.
  uint8_t* Serialize(uint8_t* p) const;

 private:
};

// Alias
typedef Value::Array Array;
typedef Value::Object Object;
typedef Value::Binary Binary;

#define GET(ctype, var)                           \
  template <>                                     \
  inline const ctype& Value::Get<ctype>() const { \
    return var;                                   \
  }                                               \
  template <>                                     \
  inline ctype& Value::Get<ctype>() {             \
    return var;                                   \
  }
GET(bool, boolean_)
GET(double, float64_)
GET(int64_t, int64_)
GET(std::string, *string_)
GET(Binary, *binary_)
GET(Array, *array_)
GET(Object, *object_)
#undef GET

// Deserialize data from memory 'p'.
// Returns error string. Empty if success.
std::string Parse(Value& v, const uint8_t* p);

class ESON {
 public:
  ESON();
  ~ESON();

  /// Load data from a file.
  bool Load(const char* filename);

  /// Dump data to a file.
  bool Dump(const char* filename);

 private:
  uint8_t* data_;  /// Pointer to data
  uint64_t size_;  /// Total data size

  bool valid_;
};
}

#endif  // __ESON_H__
