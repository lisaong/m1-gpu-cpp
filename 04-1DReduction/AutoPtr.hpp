#pragma once

template <class T> struct AutoPtr
{
    AutoPtr(T *ptr) : ptr(ptr) {}
    AutoPtr(const AutoPtr<T> &other) : ptr(other.ptr) { ptr->retain(); }

    ~AutoPtr() { ptr->release(); }
    T *operator->() { return ptr; }
    T *get() { return ptr; }

private:
    T *ptr;
};