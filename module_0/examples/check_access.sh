#!/bin/bash
# ME-5475 · check_access.sh -- run this on ARCC if you cannot find the course files.
#   bash /project/me5475/examples/check_access.sh
# Prints exactly what is wrong, and what to do about it.

echo "ME-5475 access check for: $(whoami) on $(hostname)"
echo

ok=1

if id -nG | tr ' ' '\n' | grep -qx me5475; then
  echo "  [ok]   you are in group me5475"
else
  echo "  [FAIL] you are NOT in group me5475"
  echo "         -> this is the problem. Email the instructor: your ARCC account"
  echo "            exists but is not attached to the course allocation yet."
  ok=0
fi

if [ -r /project/me5475/examples/hello.sbatch ]; then
  echo "  [ok]   you can read /project/me5475/examples"
else
  echo "  [FAIL] you cannot read /project/me5475/examples"
  [ "$ok" = 1 ] && echo "         -> unexpected; send this output to the instructor."
  ok=0
fi

if [ -d "$HOME/ME5475-examples" ]; then
  echo "  [ok]   you already have a copy at ~/ME5475-examples"
else
  echo "  [--]   no copy in your home yet. That is normal until you run the copy step."
fi

echo
if [ "$ok" = 1 ]; then
  echo "  Everything is fine. Get your own copy with:"
  echo "      cp -r /project/me5475/examples ~/ME5475-examples"
  echo "      cd ~/ME5475-examples && ls"
else
  echo "  While that is being sorted out you can still get the files from GitHub,"
  echo "  which needs no allocation and no password:"
  echo "      cd ~ && git clone https://github.com/me5475-uwyo/me5475-course.git"
  echo "      cd me5475-course/module_0/examples"
fi
echo
echo "  NOTE: the course files are NOT in your home directory by default, and they"
echo "  are not in any folder called ME5475 until you create one with the copy above."
