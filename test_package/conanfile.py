from conan import ConanFile
from conan.tools.build import can_run
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout


class CvkitTestPackageConan(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = ()

    def requirements(self):
        self.requires(self.tested_reference_str)

    def layout(self):
        cmake_layout(self)

    def generate(self):
        deps = CMakeDeps(self)
        deps.generate()

        toolchain = CMakeToolchain(self)
        toolchain.cache_variables["CMAKE_EXPORT_COMPILE_COMMANDS"] = True
        toolchain.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def test(self):
        if can_run(self):
            cmake = CMake(self)
            cmake.test()
