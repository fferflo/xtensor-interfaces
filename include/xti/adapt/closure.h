#pragma once

namespace xti {

template <typename TObject>
class Closure
{
protected:
  TObject m_object;

public:
  Closure(TObject&& object)
    : m_object(static_cast<TObject&&>(object))
  {
  }

  Closure(const Closure<TObject>& other)
    : m_object(other.m_object)
  {
  }

  Closure(Closure<TObject>&& other)
    : m_object(static_cast<TObject&&>(other.m_object))
  {
  }

  Closure<TObject>& operator=(const Closure<TObject>& other)
  {
    this->m_object = other.m_object;
    return *this;
  }

  Closure<TObject>& operator=(Closure<TObject>&& other)
  {
    this->m_object = static_cast<TObject&&>(other.m_object);
    return *this;
  }

  ~Closure()
  {
  }
};

} // end of ns xti
