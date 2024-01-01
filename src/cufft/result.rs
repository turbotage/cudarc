use super::sys::{self, cufftType, cufftReal, cufftComplex, cufftDoubleReal, cufftDoubleComplex};
use core::ffi::{c_int, c_longlong, c_void};
use core::mem::MaybeUninit;
use std::process::id;

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

/// [nvidia docs](https://docs.nvidia.com/cuda/cufft/index.html#cufftsetstream)
pub unsafe fn set_stream(
    handle: sys::cufftHandle,
    stream: sys::cudaStream_t,
) -> Result<(), CublasError> {
    sys::cufftSetStream(handle, stream).result()
}

/// [nvidia docs](https://docs.nvidia.com/cuda/cufft/index.html#cufftplan1d)
#[allow(clippy::too_many_arguments)]
pub unsafe fn plan_1d(
    plan: sys::cufftHandle,
    nx: c_int,
    type_: cufftType,
    batch: c_int,
) -> Result<(), CufftError> {
    sys::cufftPlan1d(plan, nx, type_, batch)
}

/// [nvidia docs](https://docs.nvidia.com/cuda/cufft/index.html#cufftplan2d)
#[allow(clippy::too_many_arguments)]
pub unsafe fn plan_2d(
    plan: sys::cufftHandle,
    nx: c_int,
    ny: c_int,
    type_: cufftType,
) -> Result<(), CufftError> {
    sys::cufftPlan2d(plan, nx, ny, type_)
}

/// [nvidia docs](https://docs.nvidia.com/cuda/cufft/index.html#cufftplan3d)
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

/// [nvidia docs](https://docs.nvidia.com/cuda/cufft/index.html#cufftplanmany)
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

/// [nvidia docs](https://docs.nvidia.com/cuda/cufft/index.html#cufftcreate)
pub fn create_plan_handle() -> Result<sys::cufftHandle, CublasError> {
    let mut handle = MaybeUninit::uninit();
    unsafe {
        sys::cufftCreate(handle.as_mut_ptr()).result()?;
        Ok(handle.assume_init())
    }
}

/// [nvidia docs](https://docs.nvidia.com/cuda/cufft/index.html#cufftdestroy)
pub unsafe fn destroy_plan_handle(handle: sys::cufftHandle) -> Result<(), CufftError> {
    sys::cufftDestroy(handle).result()
}

/// [nvidia docs](https://docs.nvidia.com/cuda/cufft/index.html#cufftmakeplan1d)
#[allow(clippy::too_many_arguments)]
pub unsafe fn make_plan_1d(
    plan: sys::cufftHandle,
    nx: c_int,
    type_: cufftType,
    batch: c_int,
    work_size: *mut usize,
) -> Result<(), CufftError> {
    sys::cufftMakePlan1d(plan, nx, type_, batch, work_size).result()
}

/// [nvidia docs](https://docs.nvidia.com/cuda/cufft/index.html#cufftmakeplan2d)
#[allow(clippy::too_many_arguments)]
pub unsafe fn make_plan_2d(
    plan: sys::cufftHandle,
    nx: c_int,
    ny: c_int,
    type_: cufftType,
    work_size: *mut usize,
) -> Result<(), CufftError> {
    sys::cufftMakePlan2d(plan, nx, ny, type_, work_size).result()
}

/// [nvidia docs](https://docs.nvidia.com/cuda/cufft/index.html#cufftmakeplan3d)
#[allow(clippy::too_many_arguments)]
pub unsafe fn make_plan_3d(
    plan: sys::cufftHandle,
    nx: c_int,
    ny: c_int,
    nz: c_int,
    type_: cufftType,
    work_size: *mut usize,
) -> Result<(), CufftError> {
    sys::cufftMakePlan3d(plan, nx, ny, nz, type_, work_size).result()
}

/// [nvidia docs](https://docs.nvidia.com/cuda/cufft/index.html#cufftmakeplanmany)
#[allow(clippy::too_many_arguments)]
pub unsafe fn make_plan_many(
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
    work_size: *mut usize,
) -> Result<(), CufftError> {
    sys::cufftMakePlanMany(plan, rank, n, inembed, istride, idist, onembed, 
        ostride, odist, type_, batch, work_size).result()
}

/// [nvidia docs](https://docs.nvidia.com/cuda/cufft/index.html#c.cufftExecC2C)
#[allow(clippy::too_many_arguments)]
pub unsafe fn exec_c2c(
    plan: sys::cufftHandle,
    idata: *mut cufftComplex,
    odata: *mut cufftComplex,
    direction: c_int,
) -> Result<(), CufftError> {
    sys::cufftExecC2C(plan, idata, odata, direction)
}

/// [nvidia docs](https://docs.nvidia.com/cuda/cufft/index.html#c.cufftExecR2C)
#[allow(clippy::too_many_arguments)]
pub unsafe fn exec_r2c(
    plan: sys::cufftHandle,
    idata: *mut cufftReal,
    odata: *mut cufftComplex,
) -> Result<(), CufftError> {
    sys::cufftExecR2C(plan, idata, odata)
}

/// [nvidia docs](https://docs.nvidia.com/cuda/cufft/index.html#c.cufftExecC2R)
#[allow(clippy::too_many_arguments)]
pub unsafe fn exec_c2r(
    plan: sys::cufftHandle,
    idata: *mut cufftComplex,
    odata: *mut cufftReal,
) -> Result<(), CufftError> {
    sys::cufftExecC2R(plan, idata, odata)
}

/// [nvidia docs](https://docs.nvidia.com/cuda/cufft/index.html#c.cufftExecZ2Z)
#[allow(clippy::too_many_arguments)]
pub unsafe fn exec_z2z(
    plan: sys::cufftHandle,
    idata: *mut cufftDoubleComplex,
    odata: *mut cufftDoubleComplex,
    direction: c_int,
) -> Result<(), CufftError> {
    sys::cufftExecZ2Z(plan, idata, odata, direction)
}

/// [nvidia docs](https://docs.nvidia.com/cuda/cufft/index.html#c.cufftExecD2Z)
#[allow(clippy::too_many_arguments)]
pub unsafe fn exec_d2z(
    plan: sys::cufftHandle,
    idata: *mut cufftDoubleReal,
    odata: *mut cufftDoubleComplex,
) -> Result<(), CufftError> {
    sys::cufftExecD2Z(plan, idata, odata)
}

/// [nvidia docs](https://docs.nvidia.com/cuda/cufft/index.html#c.cufftExecD2Z)
#[allow(clippy::too_many_arguments)]
pub unsafe fn exec_z2d(
    plan: sys::cufftHandle,
    idata: *mut cufftDoubleComplex,
    odata: *mut cufftDoubleReal,
) -> Result<(), CufftError> {
    sys::cufftExecZ2D(plan, idata, odata)
}




