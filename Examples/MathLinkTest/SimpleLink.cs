/********************************************************

SimpleLink.cs

A very simple .NET/Link example program demonstrating various methods from the
IKernelLink interface.

To compile this program, see the ReadMe.html file that accompanies it.

************************************************************/

using System;
using Wolfram.NETLink;
 
 public class SimpleLink {

     public static void Main(String[] args) {

        String[] mathArgs = { @"-linkname",
                              @"C:\Program Files\Wolfram Research\Mathematica\11.0\MathKernel.exe -mathlink" };

        Console.WriteLine(mathArgs.ToString());

         // This launches the Mathematica kernel:
        IKernelLink ml = MathLinkFactory.CreateKernelLink(mathArgs);
         
         // Discard the initial InputNamePacket the kernel will send when launched.
         ml.WaitAndDiscardAnswer();
         
         // Now compute 2+2 in several different ways.
         
         // The easiest way. Send the computation as a string and get the result in a single call:
         string result = ml.EvaluateToOutputForm("2+2", 0);
         Console.WriteLine("2 + 2 = " + result);
         
         // Use Evaluate() instead of EvaluateToXXX() if you want to read the result as a native type
         // instead of a string.
         ml.Evaluate("2+2");
         ml.WaitForAnswer();
         int intResult = ml.GetInteger();
         Console.WriteLine("2 + 2 = " + intResult);
         
         // You can also get down to the metal by using methods from IMathLink:
         ml.PutFunction("EvaluatePacket", 1);
         ml.PutFunction("Plus", 2);
         ml.Put(2);
         ml.Put(2);
         ml.EndPacket();
         ml.WaitForAnswer();
         intResult = ml.GetInteger();
         Console.WriteLine("2 + 2 = " + intResult);
         
         // Always Close link when done:
         ml.Close();
         
         // Wait for user to close window.
         Console.WriteLine("Press Return to exit...");
         Console.Read();
     }

 }
