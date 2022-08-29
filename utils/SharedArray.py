from typing import List, Any, TYPE_CHECKING
import numpy as np
from ctypes import c_uint8, sizeof
from typing import Union, Tuple, List
import ctypes as ct
import os
import secrets
from numpy import ndarray, prod
from .shared import SharedMemory, write_data, read_data
from .shared_memory_record import SharedMemoryRecorder

base_dir = os.path.abspath(os.path.dirname(__file__))
_SHM_SAFE_NAME_LENGTH = 14
_SHM_NAME_PREFIX = 'wnsm_'


def make_filename():
    "Create a random filename for the shared memory object."
    # number of random bytes to use for name
    nbytes = (_SHM_SAFE_NAME_LENGTH - len(_SHM_NAME_PREFIX)) // 2
    assert nbytes >= 2, '_SHM_NAME_PREFIX too long'
    name = _SHM_NAME_PREFIX + secrets.token_hex(nbytes)
    assert len(name) <= _SHM_SAFE_NAME_LENGTH
    return name


def _make_filename():
    "Create a random filename for the shared memory object."
    # number of random bytes to use for name
    nbytes = (_SHM_SAFE_NAME_LENGTH - len(_SHM_NAME_PREFIX)) // 2
    assert nbytes >= 2, '_SHM_NAME_PREFIX too long'
    name = _SHM_NAME_PREFIX + secrets.token_hex(nbytes)
    assert len(name) <= _SHM_SAFE_NAME_LENGTH
    return name


class _Nop:
    def __init__(self, *args, **kwargs):
        pass


class NDArray(ndarray):
    def set_data(self, data):
        write_data(self, data)

    def get_data(self):
        return read_data(self)

    @property
    def value(self):
        return self[0]

    @value.setter
    def value(self, val):
        self[0] = val


class Array(NDArray, SharedMemoryRecorder):
    """
    无论是windows还是linux共享内存的初始值均为0，所以无需后续的初始化值
    """

    def __new__(cls, shape, dtype: Union[str, np.dtype, object] = None, name=None, create=True, offset=0,
                strides=None, order=None, record=True):
        if not isinstance(dtype, np.dtype):
            dtype = np.dtype(dtype)
        size = int(np.prod(shape) * dtype.itemsize)
        buf = SharedMemory(name, create=create, size=size)
        obj = super().__new__(cls, shape, dtype, buf.buf, offset, strides, order)
        obj.buf = buf
        obj.record_ = record
        if create and record:
            cls.save_sm_name(buf.name, buf.size)
        return obj

    @property
    def name(self):
        if hasattr(self, "buf"):
            return self.buf.name
        raise Exception("此数组发生了转移/copy")

    @name.setter
    def name(self, val):
        raise Exception("Unsupported set name")

    def close(self):
        if not hasattr(self, "buf"):
            raise Exception("此数组发生了转移/copy")
        self.buf.close()
        self.buf.unlink()
        if self.record_:
            self.remove_sm_name(self.name)

    def release(self):
        self.close()

    def __reduce__(self):
        return Array, (self.shape, self.dtype, self.name, False)

    def copy(self, order='C'):
        # return np.ndarray(self.shape, self.dtype, self.data).copy(order)
        return np.copy(self.data, order)

    # def __del__(self):
    #     self.close()


class Value:
    def __init__(self, c_type, value=0, name=None, create=None, record=True):
        self.data = Array(1, c_type, name, create, record=record)
        self.data[0] = value

    @property
    def value(self):
        return self.data[0]


def zeros_like(a: np.ndarray, dtype=None, name=None, create=True, record=True):
    dtype = dtype or a.dtype
    shape = a.shape
    return Array(shape, dtype, name, create, record=record)


def zeros(shape, dtype=None, name=None, create=True, record=True):
    dtype = dtype or np.uint8
    return Array(shape, dtype, name, create, record=record)


def full_like(a, fill_value, dtype=None, name=None, create=True, record=True):
    dtype = dtype or a.dtype
    shape = a.shape
    arr = Array(shape, dtype, name, create, record=record)
    arr[:] = fill_value
    return arr


def full(shape, fill_value, dtype=None, name=None, create=True, record=True):
    dtype = dtype or np.uint8
    arr = Array(shape, dtype, name, create, record=record)
    arr[:] = fill_value
    return arr


def ones(shape, dtype=None, name=None, create=True, record=True):
    return full(shape, 1, dtype, name, create, record=record)


def ones_like(arr, dtype=None, name=None, create=True, record=True):
    return full_like(arr, 1, dtype, name, create, record=record)


class Close:
    def close(self):
        for key, value in self.__dict__.items():
            if isinstance(value, Array):
                value.close()
                print("name:", key, "closed.")
        print("class:", self.__class__, "closed.")


class Dict(object):

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __getattr__(self, item):
        return object.__getattribute__(self, item)

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, value):
        self.__dict__[key] = value


class FieldMeta(type):
    def __new__(mcs, what: str, bases, attr_dict):
        bases = (*filter(lambda t: t not in (NDArray,), bases),)
        cls = super().__new__(mcs, what, bases, attr_dict)
        return cls


class SharedField(NDArray, metaclass=FieldMeta):
    def __init__(self, shape: Union[List, Tuple, int], c_type: Any = c_uint8, value=0):
        object.__init__(self)
        if c_type is int:
            c_type = ct.c_int
        if c_type is float:
            c_type = ct.c_float
        setattr(self, "#name", None)
        setattr(self, "#shape", shape)
        setattr(self, "#c_type", c_type)
        setattr(self, "#value", value)


class SharedFieldUint8(SharedField):
    def __init__(self, shape, value=0):
        super().__init__(shape, c_uint8, value)


class SharedFieldInt(SharedField):
    def __init__(self, shape, value=0):
        super().__init__(shape, ct.c_int, value)


class SharedFieldInt64(SharedField):
    def __init__(self, shape, value=0):
        super().__init__(shape, ct.c_int64, value)


class SharedFieldInt32(SharedField):
    def __init__(self, shape, value=0):
        super().__init__(shape, ct.c_int32, value)


def create_shared(self, name, create, fields: List[SharedField]):
    total_size = 0
    for field in fields:
        shape = getattr(field, "#shape")
        c_type = getattr(field, "#c_type")
        total_size += prod(shape) * sizeof(c_type)
    buf = zeros(total_size, c_uint8, name, create)
    setattr(self, "$buf", buf)
    setattr(self, "$name", buf.name)
    setattr(self, "$fields", fields)
    buffer = buf.buf.buf
    offset = 0
    for field in fields:
        name = getattr(field, "#name")
        shape = getattr(field, "#shape")
        c_type = getattr(field, "#c_type")
        value = getattr(field, "#value")
        arr = NDArray(shape, c_type, buffer, offset=offset)
        if create:
            arr[:] = value
        setattr(self, name, arr)
        offset += prod(shape) * sizeof(c_type)
    return self


class SharedStructureMeta(type):
    def __call__(cls, *args, **kwargs):
        self = super().__call__(*args, **kwargs)
        name = getattr(self, "$name")
        create = getattr(self, "$create")
        fields: List[SharedField] = []
        for k, v in vars(self).items():
            if isinstance(v, SharedField):
                if k in {"get_sm_name", "close"}:
                    raise Exception("Field name error")
                setattr(v, "#name", k)
                fields.append(v)
        create_shared(self, name, create, fields)
        return self


class SharedStructure(metaclass=SharedStructureMeta):

    def __init__(self, name=None, create=True):
        setattr(self, "$name", name)
        setattr(self, "$create", create)

    def get_sm_name(self):
        return getattr(self, "$buf").name

    def close(self):
        return getattr(self, "$buf").close()

    def __getstate__(self):
        return self.get_sm_name(), getattr(self, "$fields")

    def __setstate__(self, state):
        name, fields = state
        SharedStructure.__init__(self, name, False)
        create_shared(self, name, False, fields)

# def task(a):
#     print(a.a)
#
#
# class Test(SharedStructure):
#     def __init__(self):
#         super().__init__("test_")
#         self.a = SharedField(8)
#         self.a.value = 12
#
#
# if __name__ == '__main__':
#     # a = Array(10, "uint8")
#     # print(a)
#     from multiprocessing import Process
#
#     a = Test()
#     print(a.a)
#     proc = Process(target=task, args=(a,))
#     proc.start()
#     proc.join()
