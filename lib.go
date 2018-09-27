// +build linux

package tensorrt

// #cgo LDFLAGS: -lnvinfer -lnvcaffe_parser -lcudart -L${SRCDIR} -lstdc++
// #cgo CXXFLAGS: -std=c++11  -O3 -Wall -g
// #cgo CXXFLAGS: -Wno-sign-compare -Wno-unused-function -I${SRCDIR}/cbits
// #cgo linux,!ppc64le CXXFLAGS: -I/usr/local/cuda/include
// #cgo linux,amd64 CXXFLAGS: -I/opt/tensorrt/include -I/usr/include/x86_64-linux-gnu
// #cgo linux,arm64 CXXFLAGS: -I/usr/include/aarch64-linux-gnu
// #cgo linux,!ppc64le  LDFLAGS: -L/usr/local/cuda/lib64 -L/opt/tensorrt/lib -L/usr/lib/x86_64-linux-gnu -lcudnn -lcudart
import "C"
