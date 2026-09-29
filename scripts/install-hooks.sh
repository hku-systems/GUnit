#!/bin/sh
# This script configures git to use the shared hooks directory.

git config core.hooksPath githooks

echo "Git hooks have been configured to use the 'githooks' directory."