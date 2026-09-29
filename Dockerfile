# Use Ubuntu as the base image
FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

# Update and install sudo
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    clang \
    llvm \
    llvm-dev \
    libssl-dev \
    wget \
    git \
    gdb \
    sudo \
    neovim \
    tmux \
    && apt-get clean

# Create a non-root user and add to the sudo group
ARG USERNAME=rapid
ARG USER_ID
ARG GROUP_ID
# RUN groupadd -g ${GROUP_ID} rapid && \
#     useradd -m -u ${USER_ID} -g ${GROUP_ID} rapid && \
#     echo "rapid:rapid" | chpasswd && \
#     adduser rapid sudo

RUN if getent group ${GROUP_ID}; then \
        echo "Group with GID ${GROUP_ID} already exists. Renaming group to '${USERNAME}.'"; \
        groupmod -n ${USERNAME} $(getent group ${GROUP_ID} | cut -d: -f1); \
    else \
        echo "Creating group '${USERNAME}' with GID ${GROUP_ID}."; \
        groupadd -g ${GROUP_ID} ${USERNAME}; \
    fi && \
    if id -u ${USER_ID} >/dev/null 2>&1; then \
        echo "User with UID ${USER_ID} already exists. Renaming user to '${USERNAME}.'"; \
        OLD_USERNAME=$(id -nu ${USER_ID}) && \
        usermod -l ${USERNAME} ${OLD_USERNAME} && \
        OLD_HOME="/home/${OLD_USERNAME}" && \
        NEW_HOME="/home/${USERNAME}" && \
        usermod -d ${NEW_HOME} ${USERNAME} && \
        # Move old home data to new home directory
        mv ${OLD_HOME} ${NEW_HOME} && \
        echo "Moved data from ${OLD_HOME} to ${NEW_HOME}."; \
    else \
        echo "Creating user '${USERNAME}' with UID ${USER_ID}."; \
        useradd -m -u ${USER_ID} -g ${GROUP_ID} ${USERNAME}; \
    fi && \
    # Set password for the user
    echo "${USERNAME}:${USERNAME}" | chpasswd && \
    # Add user to the sudo group
    echo "${USERNAME} ALL=(ALL:ALL) ALL" >> /etc/sudoers


# Set the default user to the specified USERNAME
USER ${USERNAME}

# Set the working directory
WORKDIR /home/${USERNAME}

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.87.0

# Set environment variable
ENV PATH="$PATH:/home/${USERNAME}/.cargo/bin"

# Start bash shell
CMD ["/bin/bash"]
