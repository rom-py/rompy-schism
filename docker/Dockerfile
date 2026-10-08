# SCHISM runtime image for rompy-schism.
#
# Builds a SCHISM release with MPI and NetCDF, as two executables:
#
#   schism             hydrodynamics
#   schism_wwm         hydrodynamics coupled with the WWM wave model
#   combine_hotstart7  merges the per-process hotstart files into hotstart.nc
#
# SCHISM always runs under MPI, with the number of scribe processes (which only
# write output) as its argument; the total number of processes includes them:
#
#   docker run --rm -v "$PWD":/data ghcr.io/rom-py/schism mpirun -n 6 schism 2
#
# Build a release (see releases.txt for the versions and checksums):
#   docker build -t schism:5.13.0 \
#     --build-arg SCHISM_VERSION=5.13.0 --build-arg SCHISM_SHA256=<sha256> docker
#
# The compiler settings are shared with the test image in docker/schism. No
# ENTRYPOINT is set so the image works with rompy's DockerConfig backend, which
# runs `bash -c "<command>"` inside the container.

ARG BASE_IMAGE=ubuntu:24.04

FROM ${BASE_IMAGE} AS builder

ARG SCHISM_VERSION=5.13.0
ARG SCHISM_SHA256
ARG SCHISM_URL=https://github.com/schism-dev/schism/archive/refs/tags

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        cmake \
        curl \
        g++ \
        gcc \
        gfortran \
        libnetcdf-dev \
        libnetcdff-dev \
        libopenmpi-dev \
        make \
        openmpi-bin \
        patch \
        perl \
        python3 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/local/bin/python

WORKDIR /usr/src/schism
# A tarball has no git history, so SCHISM gets its version label from
# src/Core/schism_version_user.txt
RUN curl -fsSL -o /tmp/schism.tar.gz "${SCHISM_URL}/v${SCHISM_VERSION}.tar.gz" \
    && if [ -n "${SCHISM_SHA256}" ]; then \
        echo "${SCHISM_SHA256}  /tmp/schism.tar.gz" | sha256sum -c -; \
    fi \
    && tar -xzf /tmp/schism.tar.gz --strip-components=1 \
    && rm /tmp/schism.tar.gz \
    && echo "v${SCHISM_VERSION}" > src/Core/schism_version_user.txt

# Fixes to the SCHISM source, see schism/patches/README.md
COPY schism/patches /tmp/patches
RUN for p in /tmp/patches/*.patch; do patch -p1 < "$p"; done

COPY schism/Make.defs.gfortran mk/Make.defs.local
COPY schism/include_modules schism/include_modules_wwm /tmp/

# -static fails against the shared MPI and NetCDF libraries of the base image
RUN sed -i 's/ -static -static-libgfortran//' mk/Make.defs.local \
    && cp /tmp/include_modules mk/include_modules \
    && make -C src -j"$(nproc)" psc \
    && install src/pschism_git_HYDRO_VL /usr/local/bin/schism

# SCHISM writes one hotstart file per process: combine_hotstart7 merges them into
# the hotstart.nc that a following run starts from
RUN cd src/Utility/Combining_Scripts \
    && gfortran -ffree-line-length-none -cpp -O2 -o /usr/local/bin/combine_hotstart7 \
        ../UtilLib/argparse.f90 ../UtilLib/schism_geometry.f90 combine_hotstart7.f90 \
        -I/usr/include -L/usr/lib/x86_64-linux-gnu -lnetcdf -lnetcdff

# WWM's Fortran module dependencies are not ordered for a parallel make
RUN make -C src clean \
    && cp /tmp/include_modules_wwm mk/include_modules \
    && make -C src psc \
    && install src/pschism_git_HYDRO_VL_WWM /usr/local/bin/schism_wwm


FROM ${BASE_IMAGE}

ARG SCHISM_VERSION=5.13.0
LABEL org.opencontainers.image.title="SCHISM ${SCHISM_VERSION}" \
      org.opencontainers.image.description="SCHISM with MPI and NetCDF, for rompy-schism" \
      org.opencontainers.image.source="https://github.com/rom-py/rompy-schism"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgfortran5 \
        libnetcdf19t64 \
        libnetcdff7 \
        openmpi-bin \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/bin/schism /usr/local/bin/schism_wwm \
    /usr/local/bin/combine_hotstart7 /usr/local/bin/

# Containers run as root; allow Open MPI to start there. Scribes mostly wait for
# output, so allow more processes than cores (mpirun would refuse otherwise)
ENV OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    OMPI_MCA_rmaps_base_oversubscribe=1

WORKDIR /data
