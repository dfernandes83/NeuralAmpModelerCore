#include "nam_file.h"

#include <array>
#include <fstream>
#include <string>

namespace nam
{
void validate_nam_json(const nlohmann::json& config)
{
  if (!config.is_object())
    throw NamFileValidationError("Invalid .nam model: root JSON value must be an object.");

  constexpr std::array<const char*, 4> required_keys{"version", "architecture", "config", "weights"};
  for (const auto* key : required_keys)
  {
    if (config.find(key) == config.end())
      throw NamFileValidationError(std::string("Invalid .nam model: missing required key \"") + key + "\".");
  }
}

nlohmann::json validate_nam_file(const std::filesystem::path& filename)
{
  if (!std::filesystem::exists(filename))
    throw NamFileValidationError("Could not validate .nam file [" + filename.string() + "]: file does not exist.");

  std::ifstream input(filename);
  if (!input.is_open())
    throw NamFileValidationError("Could not validate .nam file [" + filename.string() + "]: file could not be read.");

  nlohmann::json config;
  try
  {
    input >> config;
  }
  catch (const nlohmann::json::parse_error& error)
  {
    throw NamFileValidationError("Could not parse .nam file [" + filename.string() + "]: " + error.what());
  }

  try
  {
    validate_nam_json(config);
  }
  catch (const NamFileValidationError& error)
  {
    throw NamFileValidationError("Could not validate .nam file [" + filename.string() + "]: " + error.what());
  }

  return config;
}
} // namespace nam
