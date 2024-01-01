#![allow(clippy::too_many_arguments)]

use super::{result, result::CufftError, sys};

use crate::{driver::{CudaDevice, CudaStream, DevicePtr, DevicePtrMut}, cublas::CudaBlas};

use core::ffi::{c_int, c_longlong};
use std::sync::Arc;


#[derive(Debug)]
pub struct CudaFFT{
    pub(crate) plan: sys::cufftHandle,
    pub(crate) device: Arc<CudaDevice>,
}

unsafe impl Send for CudaFFT{}
unsafe impl Sync for CudaFFT{}

impl CudaFFT {

    pub fn new(device: Arc<CudaDevice>) -> Result<self, CufftError> {
        device.bind_to_thread().unwrap();
        let plan = result::create_plan_handle();
        let fft = self { plan, device};
        unsafe { result::set_stream(plan, fft.device.stream as *mut _) }?;
        Ok(fft)
    }

    pub fn handle(&self) -> &sys::cufftHandle {
        &self.handle
    }

    pub unsafe fn set_stream(&self, opt_stream: Option<&CudaStream>) -> Result<(), CufftError> {
        match opt_stream {
            Some(s) => result::set_stream(self.plan, s.stream as *mut _),
            None => result::set_stream(self.plan, self.device.stream as *mut _)
        }
    }

    pub fn new_plan_1d(&self, nx: c_int,  )


}
        

impl Drop for CudaBlas {
    fn drop(&mut self) {
        let handle = std::mem::replace(&mut self.handle, std::ptr::null_mut());
        if !handle.is_null() {
            unsafe { result::destroy_plan_handle(handle)}.unwrap();
        }
    }
}

#[derive(Debug, Copy, Clone)]

