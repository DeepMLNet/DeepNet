$doctemp = "$env:TEMP\DeepNetDocs"
if (Test-Path $doctemp) { rm -Recurse -Force $doctemp }
mkdir $doctemp
pushd $doctemp
git clone -b gh-pages https://github.com/DeepMLNet/DeepNet.git .
rm -Recurse *
popd

.\Generate-Docs.ps1
cp -Recurse docs\output\* $doctemp\

pushd $doctemp
echo "www.deepml.net" > CNAME
git add --all .
git commit -m "automatic documentation generation"
git push
popd

