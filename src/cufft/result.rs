use super::sys;
use core::ffi::{c_int, c_longlong, c_void};
use core::mem::MaybeUninit;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CufftError(pub sys::cufftResult_t);

impl sys::cufftResult_t {
    pub fn result(self) -> Result<(), CufftError> {
        match self {
            sys::cufftResult_t::CUFFT_SUCCESS => Ok(()),
            _ => Err(CufftError(self))
        }
    }
}

#[cfg(feature = "std")]
impl std::fmt::Display for CufftError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CufftError {}

/// [nvidia docs](https://docs.nvidia.com/cuda/cufft/index.html#cufftcreate)
pub fn create_handle() -> Result<sys::cufftHandle, CublasError> {
    let mut handle = MaybeUninit::uninit();
    unsafe {
        sys::cufftCreate(handle.as_mut_ptr()).result()?;
        Ok(handle.assume_init())
    }
}

/// [nvidia docs](https://docs.nvidia.com/cuda/cufft/index.html#cufftdestroy)
pub unsafe fn destroy_handle(handle: sys::cufftHandle) -> Result<(), CufftError> {
    sys::cufftDestroy(handle).result()
}

/// [nvidia docs](https://docs.nvidia.com/cuda/cufft/index.html#cufftsetstream)
pub unsafe fn set_stream(
    handle: sys::cufftHandle,
    stream: sys::cudaStream_t,
) -> Result<(), CublasError> {
    sys::cufftSetStream(handle, stream).result()
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn plan_1d(
    plan: sys::cufftHandle,
    nx: c_int,
    type_: cufftType,
) -> Result<(), CufftError> {
    sys::cufftPlan1d(plan, nx, type_)
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn plan_2d(
    plan: sys::cufftHandle,
    nx: c_int,
    ny: c_int,
    type_: cufftType,
) -> Result<(), CufftError> {
    sys::cufftPlan2d(plan, nx, ny, type_)
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn plan_3d(
    plan: sys::cufftHandle,
    nx: c_int,
    ny: c_int,
    nz: c_int,
    type_: cufftType,
) -> Result<(), CufftError> {
    sys::cufftPlan3d(plan, nx, ny, nz, type_)
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn plan_many(
    plan: sys::cufftHandle,
    rank: c_int,
    n: *mut c_int,
    inembed: *mut c_int,
    istride: c_int,
    idist: c_int,
    onembed: *mut c_int,
    ostride: c_int,
    odist: c_int,
    type_: cufftType,
    batch: c_int,
) -> Result<(), CufftError> {
    sys::cufftPlanMany(handle, rank, n, inembed, istride, idist,
        onembed, ostride, odist, type_, batch)
}







