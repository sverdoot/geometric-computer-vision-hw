DATAPATH="validation"
OUTPATH="out"

mkdir -p $OUTPATH

if [ -f "$DATAPATH" ] ; then
    wget https://www.dropbox.com/s/lxg7lb8xqcmxowa/validation.zip?dl=0
    unzip -o 'validation.zip?dl=0'
    rm 'validation.zip?dl=0'
    rm -r __MACOSX
fi