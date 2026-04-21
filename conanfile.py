from conan import ConanFile
from conan.tools.build import check_min_cppstd
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain
import os


class CvkitConan(ConanFile):
    name = "cvkit"
    version = "0.1.0"
    package_type = "library"
    test_type = "explicit"

    license = "Proprietary"
    author = "cvkit"
    url = "https://example.invalid/cvkit"
    description = "Computer vision toolkit"
    topics = ("cpp", "cmake", "conan", "vision")

    settings = "os", "compiler", "build_type", "arch"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
    }
    default_options = {
        "shared": True,
        "fPIC": True,
        "basekit/*:shared": True,
    }

    exports_sources = (
        "CMakeLists.txt",
        "cmake/*",
        "include/*",
        "src/*",
        "examples/*",
        "tests/*",
        "benchmark/*",
    )

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def configure(self):
        if self.options.shared:
            self.options.rm_safe("fPIC")

    def layout(self):
        self.folders.source = "."
        build_type = str(self.settings.build_type)
        self.folders.build = os.path.join("conan", build_type)
        self.folders.generators = os.path.join("conan", build_type, "generators")
        self.cpp.source.includedirs = ["include"]

        source_include_dir = os.path.join(self.recipe_folder, "include")
        generated_includedirs = [
            "src/core/generated",
            "src/media/generated",
            "src/image/generated",
            "src/infer/generated",
        ]

        self.cpp.build.includedirs = [source_include_dir, *generated_includedirs]
        self.cpp.build.libdirs = ["lib"]
        self.cpp.build.bindirs = ["bin"]

        for component in ("core", "media", "image", "infer"):
            self.cpp.build.components[component].includedirs = [
                source_include_dir,
                f"src/{component}/generated",
            ]
            self.cpp.build.components[component].libdirs = ["lib"]
            self.cpp.build.components[component].bindirs = ["bin"]

    def requirements(self):
        self.requires("basekit/0.1.0")

    def build_requirements(self):
        self.test_requires("catch2/[>=3 <4]")
        self.test_requires("benchmark/[>=1.8 <2]")

    def validate(self):
        check_min_cppstd(self, 17)

    def generate(self):
        deps = CMakeDeps(self)
        deps.generate()

        toolchain = CMakeToolchain(self)
        toolchain.cache_variables["CMAKE_EXPORT_COMPILE_COMMANDS"] = True
        toolchain.variables["BUILD_SHARED_LIBS"] = self.options.shared
        toolchain.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        self.cpp_info.set_property("cmake_file_name", "cvkit")
        self.cpp_info.set_property("cmake_target_name", "cvkit::cvkit")

        self.cpp_info.components["core"].set_property("cmake_target_name", "cvkit::core")
        self.cpp_info.components["core"].libs = ["cvkit_core"]
        self.cpp_info.components["core"].requires = ["basekit::core"]

        self.cpp_info.components["media"].set_property("cmake_target_name", "cvkit::media")
        self.cpp_info.components["media"].libs = ["cvkit_media"]
        self.cpp_info.components["media"].requires = ["core", "basekit::log"]

        self.cpp_info.components["image"].set_property("cmake_target_name", "cvkit::image")
        self.cpp_info.components["image"].libs = ["cvkit_image"]
        self.cpp_info.components["image"].requires = ["core"]

        self.cpp_info.components["infer"].set_property("cmake_target_name", "cvkit::infer")
        self.cpp_info.components["infer"].libs = ["cvkit_infer"]
        self.cpp_info.components["infer"].requires = [
            "core",
            "image",
            "basekit::config",
            "basekit::log",
        ]
