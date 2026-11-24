// Linux / ELF / GCC-or-Clang constructor interposer for NCrystal <= 4.4.6.
// This is a separate shared library; it does not change NCrystal or Liquids.
#include "NCrystal/interfaces/NCSABData.hh"
#include "NCrystal/internal/sabscatter/NCSABScatter.hh"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace NC = NCrystal;
namespace {
using Ctor = void (*)(NC::SABScatter*, NC::SABData&&, const NC::VectD&);
std::atomic<unsigned> next_id{0};

void validate_finite_scalar(double x, const char* name)
{
  // Reject non-finite values immediately so corrupted physics never enters binary dumps.
  if (!std::isfinite(x))
    throw std::runtime_error(std::string("non-finite scalar in SABData: ") + name);
}

void validate_finite_vector(const NC::VectD& v, const char* name)
{
  for (std::size_t i = 0; i < v.size(); ++i) {
    if (!std::isfinite(v[i]))
      throw std::runtime_error(std::string("non-finite array value in SABData: ") + name);
  }
}

void write_binary_vector(std::ostream& out, const NC::VectD& v)
{
  out.write(reinterpret_cast<const char*>(v.data()),
            static_cast<std::streamsize>(v.size() * sizeof(double)));
  if (!out)
    throw std::runtime_error("failed writing binary array payload");
}

unsigned dump(const NC::SABData& d, const char* dir)
{
  const auto& a = d.alphaGrid();
  const auto& b = d.betaGrid();
  const auto& s = d.sab();
  if (a.empty() || b.empty() || s.size() != a.size() * b.size())
    throw std::runtime_error("inconsistent SABData dimensions");

  const double temp_k = d.temperature().dbl();
  const double bound_xs = d.boundXS().dbl();
  const double mass_amu = d.elementMassAMU().dbl();
  const double suggested_emax = d.suggestedEmax();

  validate_finite_scalar(temp_k, "temperature_K");
  validate_finite_scalar(bound_xs, "bound_xs_barn");
  validate_finite_scalar(mass_amu, "element_mass_amu");
  validate_finite_scalar(suggested_emax, "suggested_emax_eV");
  validate_finite_vector(a, "alpha");
  validate_finite_vector(b, "beta");
  validate_finite_vector(s, "sab");

  const unsigned id = next_id.fetch_add(1);
  std::ostringstream stem;
  stem << dir << "/sab_" << std::setfill('0') << std::setw(4) << id;

  // Write raw IEEE-754 float64 arrays to a binary file to avoid multi-megabyte ASCII
  // formatting and parsing overhead across large parallel ensembles.
  const std::string bin_path = stem.str() + ".bin";
  std::ofstream bin_out(bin_path, std::ios::binary | std::ios::trunc);
  if (!bin_out)
    throw std::runtime_error("could not open " + bin_path);
  write_binary_vector(bin_out, a);
  write_binary_vector(bin_out, b);
  write_binary_vector(bin_out, s);
  bin_out.close();
  if (!bin_out)
    throw std::runtime_error("failed closing " + bin_path);

  // Write compact JSON metadata referencing the binary payload.
  const std::string json_path = stem.str() + ".json";
  std::ofstream out(json_path, std::ios::trunc);
  if (!out)
    throw std::runtime_error("could not open " + json_path);
  out << std::setprecision(17);
  out << "{\n"
      << "  \"temperature_K\": " << temp_k << ",\n"
      << "  \"bound_xs_barn\": " << bound_xs << ",\n"
      << "  \"element_mass_amu\": " << mass_amu << ",\n"
      << "  \"suggested_emax_eV\": " << suggested_emax << ",\n"
      << "  \"n_alpha\": " << a.size() << ",\n"
      << "  \"n_beta\": " << b.size() << ",\n"
      << "  \"binary_file\": \"" << ("sab_" + stem.str().substr(stem.str().rfind('_') + 1) + ".bin") << "\",\n"
      << "  \"layout\": \"sab[beta_index][alpha_index]\"\n"
      << "}\n";
  out.close();
  if (!out)
    throw std::runtime_error("failed closing " + json_path);
  return id;
}

void dump_component_xs(NC::SABScatter* self, const char* dir, unsigned id)
{
  std::ifstream energies_in(std::string(dir) + "/energies.bin", std::ios::binary);
  std::vector<double> energies;
  if (energies_in) {
    energies_in.seekg(0, std::ios::end);
    const std::streamsize nbytes = energies_in.tellg();
    energies_in.seekg(0, std::ios::beg);
    if (nbytes <= 0 || nbytes % static_cast<std::streamsize>(sizeof(double)) != 0)
      throw std::runtime_error("invalid energies.bin size");
    energies.resize(static_cast<std::size_t>(nbytes) / sizeof(double));
    energies_in.read(reinterpret_cast<char*>(energies.data()), nbytes);
  } else {
    // Fallback for text energy grids if invoked by external diagnostic tools.
    std::ifstream txt_in(std::string(dir) + "/energies.txt");
    if (!txt_in)
      throw std::runtime_error("missing energies.bin or energies.txt");
    double e = 0.0;
    while (txt_in >> e)
      energies.push_back(e);
  }

  std::vector<double> xs_values(energies.size());
  NC::CachePtr cache;
  for (std::size_t i = 0; i < energies.size(); ++i) {
    const double val = self->crossSectionIsotropic(cache, NC::NeutronEnergy{energies[i]}).dbl();
    validate_finite_scalar(val, "component_xs");
    xs_values[i] = val;
  }

  std::ostringstream bin_name;
  bin_name << dir << "/xs_" << std::setfill('0') << std::setw(4) << id << ".bin";
  std::ofstream xs_bin(bin_name.str(), std::ios::binary | std::ios::trunc);
  if (!xs_bin)
    throw std::runtime_error("could not open " + bin_name.str());
  xs_bin.write(reinterpret_cast<const char*>(xs_values.data()),
               static_cast<std::streamsize>(xs_values.size() * sizeof(double)));
  xs_bin.close();
  if (!xs_bin)
    throw std::runtime_error("failed writing component cross sections");
}

void forward(const char* sym, NC::SABScatter* self,
             NC::SABData&& sab, const NC::VectD& egrid)
{
  const char* dir = std::getenv("NCRYSTAL_SAB_DUMP_DIR");
  const bool capture = dir && *dir;
  unsigned id = 0;
  try {
    if (capture)
      id = dump(sab, dir);
  } catch (const std::exception& e) {
    std::cerr << "[sabtap] capture error before constructor: " << e.what() << '\n';
    std::abort();
  }

  dlerror();
  auto* addr = dlsym(RTLD_NEXT, sym);
  if (const char* error = dlerror()) {
    std::cerr << "[sabtap] cannot forward " << sym << ": " << error << '\n';
    std::abort();
  }
  reinterpret_cast<Ctor>(addr)(self, std::move(sab), egrid);

  try {
    if (capture)
      dump_component_xs(self, dir, id);
  } catch (const std::exception& e) {
    std::cerr << "[sabtap] cross-section evaluation error: " << e.what() << '\n';
    std::abort();
  }
}
}  // namespace

// Itanium C++ ABI symbols for NCrystal::SABScatter::SABScatter(
//     NCrystal::SABData&&, const std::vector<double>&).
// Both C1 (complete object) and C2 (base object) variants are covered.
extern "C" void sabtap_ctor_C1(NC::SABScatter*, NC::SABData&&, const NC::VectD&)
  __asm__("_ZN8NCrystal10SABScatterC1EONS_7SABDataERKSt6vectorIdSaIdEE");
extern "C" void sabtap_ctor_C2(NC::SABScatter*, NC::SABData&&, const NC::VectD&)
  __asm__("_ZN8NCrystal10SABScatterC2EONS_7SABDataERKSt6vectorIdSaIdEE");

extern "C" void sabtap_ctor_C1(NC::SABScatter* self, NC::SABData&& sab,
                               const NC::VectD& egrid)
{
  forward("_ZN8NCrystal10SABScatterC1EONS_7SABDataERKSt6vectorIdSaIdEE",
          self, std::move(sab), egrid);
}
extern "C" void sabtap_ctor_C2(NC::SABScatter* self, NC::SABData&& sab,
                               const NC::VectD& egrid)
{
  forward("_ZN8NCrystal10SABScatterC2EONS_7SABDataERKSt6vectorIdSaIdEE",
          self, std::move(sab), egrid);
}