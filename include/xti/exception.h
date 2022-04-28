#pragma once

#include <exception>
#include <stdexcept>
#include <string>
#include <filesystem>

namespace xti {

class InvalidDimensionException : public std::exception
{
public:
  InvalidDimensionException(std::string message)
    : m_message(message)
  {
  }

  virtual const char* what() const throw ()
  {
    return m_message.c_str();
  }

private:
  std::string m_message;
};

class InvalidLayoutException : public std::exception
{
public:
  InvalidLayoutException(std::string message)
    : m_message(message)
  {
  }

  InvalidLayoutException()
    : InvalidLayoutException("Layout not supported")
  {
  }

  virtual const char* what() const throw ()
  {
    return m_message.c_str();
  }

private:
  std::string m_message;
};

class InvalidElementTypeException : public std::exception
{
public:
  InvalidElementTypeException(std::string message)
    : m_message(message)
  {
  }

  InvalidElementTypeException(size_t got_size, size_t expected_size)
    : InvalidElementTypeException("Invalid element size. Got " + std::to_string(got_size) + ", expected " + std::to_string(expected_size))
  {
  }

  virtual const char* what() const throw ()
  {
    return m_message.c_str();
  }

private:
  std::string m_message;
};

class ResizeNotSupportedException : public std::exception
{
public:
  ResizeNotSupportedException(std::string message)
    : m_message(message)
  {
  }

  ResizeNotSupportedException()
  {
  }

  virtual const char* what() const throw ()
  {
    return m_message.c_str();
  }

private:
  std::string m_message;
};

} // end of ns xti
