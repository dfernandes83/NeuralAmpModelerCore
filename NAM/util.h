#pragma once

// Utilities

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <Eigen/Dense> // Eigen::MatrixXf

namespace nam
{
namespace util
{
/// \brief Convert a string to lowercase
/// \param s Input string
/// \return Lowercase version of the input string
std::string lowercase(const std::string& s);

/// \brief A bounds-checked cursor over a flat weights vector, consumed sequentially while
/// constructing a model's layers from its .nam file.
///
/// Every architecture's weight-loading code walks a flat `std::vector<float>` of weights,
/// historically via a raw `std::vector<float>::iterator` advanced with `*(weights++)` and no
/// check against the end of the vector. A truncated or corrupted .nam file -- fewer weights than
/// the architecture's config implies it needs -- makes that iterator walk past the end of the
/// underlying allocation: undefined behavior (heap over-read), not a catchable error. A `try`/
/// `catch` around the whole load can't make undefined behavior recoverable, only a real
/// exception.
///
/// WeightCursor::Next() does the same "read current, advance" as `*(weights++)`, but throws
/// `std::runtime_error` instead of ever dereferencing past the end. AtEnd()/Remaining() let a
/// caller check the same "did we consume exactly all the weights, no more, no less" contract
/// some architectures already checked by hand with a second iterator kept in sync manually.
class WeightCursor
{
public:
  WeightCursor(std::vector<float>::const_iterator begin, std::vector<float>::const_iterator end)
  : _it(begin), _end(end)
  {
  }
  explicit WeightCursor(const std::vector<float>& weights) : _it(weights.cbegin()), _end(weights.cend()) {}

  /// \brief Read the next weight and advance.
  /// \throws std::runtime_error if the cursor is already at the end (too few weights provided).
  float Next()
  {
    if (_it == _end)
      throw std::runtime_error(
        "Model weights ended unexpectedly while loading -- the file may be truncated or corrupted.");
    return *(_it++);
  }

  /// \brief True once every weight up to the end has been consumed.
  bool AtEnd() const { return _it == _end; }

  /// \brief How many weights remain unconsumed.
  std::vector<float>::difference_type Remaining() const { return _end - _it; }

private:
  std::vector<float>::const_iterator _it;
  std::vector<float>::const_iterator _end;
};

/// \brief Throws std::runtime_error if `value` isn't in (0, maxValue] -- guards architecture
/// dimensions (layer counts, channel counts, kernel sizes, ...) parsed straight from a .nam
/// file's JSON before they feed into allocation-sizing arithmetic (e.g. `4 * hidden_size`,
/// `(kernel_size - 1) * dilation`). A corrupted or hostile file could otherwise supply a negative
/// or absurdly large value, causing signed-integer overflow (undefined behavior) or an attempt to
/// allocate an unreasonable amount of memory before any exception has a chance to fire.
/// \param value The parsed dimension to check
/// \param maxValue Generous upper bound -- see call sites for the reasoning behind each limit
/// \param name Human-readable name of the field, used in the exception message
inline void CheckDimension(const long long value, const long long maxValue, const char* name)
{
  if (value <= 0 || value > maxValue)
    throw std::runtime_error(std::string("Model config field '") + name + "' has an out-of-range value ("
                              + std::to_string(value) + ") -- the file may be corrupted.");
}
}; // namespace util
}; // namespace nam
