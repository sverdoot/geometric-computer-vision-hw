DATAPATH="validation"

mkdir -p out/med/bilin
mkdir -p out/high/bilin

mkdir -p out/med/bispline
mkdir -p out/med/bispline

if ! [ -d "$DATAPATH" ] ; then
    wget https://www.dropbox.com/s/lxg7lb8xqcmxowa/validation.zip?dl=0
    unzip -o 'validation.zip?dl=0'
    rm 'validation.zip?dl=0'
    rm -r __MACOSX
fi