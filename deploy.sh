#!/bin/bash

# Push NuGet packages.
dotnet nuget push -s https://www.myget.org/F/coreports/api/v2/package -k $MYGETKEY 'Packages/Release/*.nupkg' || true

# Push to Tensor.Sample repository.
if [ "$TRAVIS_BRANCH" = "master" ]; then 
    git clone https://${GITHUB_TOKEN}@github.com/DeepMLNet/Tensor.Sample.git sampleDeploy_old
    cp -av Tensor/Tensor.Sample sampleDeploy
    mv sampleDeploy_old/.git sampleDeploy/
    cd sampleDeploy
    rm -f Tensor.Sample.Internal.fsproj
    git add .
    git commit -m "CI build"
    git push 
fi
