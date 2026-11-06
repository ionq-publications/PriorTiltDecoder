fn main() {
    // Linker flags a Python extension module needs (e.g. `-undefined dynamic_lookup` on macOS).
    pyo3_build_config::add_extension_module_link_args();
}
