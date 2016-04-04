require 'nn'
require 'rnn'
------------------------------------------------------------------------
--[[ Repeater ]]--
-- Encapsulates an AbstractRecurrent instance (rnn) which is repeatedly 
-- presented with the same input for rho time steps.
-- The output is a table of rho outputs of the rnn.
------------------------------------------------------------------------
local RCLayer, parent = torch.class('nn.RCLayer', 'nn.Repeater')


-- modify by patrick
function RCLayer:updateOutput(input)
   self.module = self.module or self.rnn -- backwards compatibility

   self.module:forget()
   -- TODO make copy outputs optional
   for step=1,self.rho do
      --print('<<<<< Step:  ', step, torch.pointer(self.module))
      self.output[step] = nn.rnn.recursiveCopy(self.output[step], self.module:updateOutput(input))
   end
   --print('===========================================\n\n')
   return self.output[self.rho]
end

-- modify by patrick
function RCLayer:updateGradInput(input, gradOutput)
   --print('========== I am updateGradInput =======================')
   assert(self.module.step - 1 == self.rho, "inconsistent rnn steps")
   --[=[
   assert(torch.type(gradOutput) == 'table', "expecting gradOutput table")
   assert(#gradOutput == self.rho, "gradOutput should have rho elements")
   
   -- back-propagate through time (BPTT)
   for step=self.rho,1,-1 do
      local gradInput = self.module:updateGradInput(input, gradOutput[step])
      if step == self.rho then
         self.gradInput = nn.rnn.recursiveCopy(self.gradInput, gradInput)
      else
         nn.rnn.recursiveAdd(self.gradInput, gradInput)
      end
   end
   ]=]

   local gradPrevOutput = gradOutput
   for step=self.rho,1,-1 do
       --print('<<<<< Step:  ', step, torch.pointer(self.module))
      local result = self.module:updateGradInput(input, gradPrevOutput)
      local gradInput = result[1]
      gradPrevOutput = result[2]
      if step == self.rho then
         self.gradInput = nn.rnn.recursiveCopy(self.gradInput, gradInput)
      else
         nn.rnn.recursiveAdd(self.gradInput, gradInput)
      end
   end

   --print('===========================================\n\n')
   return self.gradInput
end

-- modify by patrick
function RCLayer:accGradParameters(input, gradOutput, scale)
   --print('========== I am accGradParmeters =======================')
   assert(self.module.step - 1 == self.rho, "inconsistent rnn steps")
   --[=[
   assert(torch.type(gradOutput) == 'table', "expecting gradOutput table")
   assert(#gradOutput == self.rho, "gradOutput should have rho elements")
   
   -- back-propagate through time (BPTT)
   for step=self.rho,1,-1 do
      self.module:accGradParameters(input, gradOutput[step], scale)
   end
   ]=]
   self.module:accGradParameters(input, gradOutput, scale)
end

-- overide by patrick
function RCLayer:accUpdateGradParameters(input, gradOutput, lr)
   assert(self.module.step - 1 == self.rho, "inconsistent rnn steps")
   --[=[
   assert(torch.type(gradOutput) == 'table', "expecting gradOutput table")
   assert(#gradOutput == self.rho, "gradOutput should have rho elements")
   
   -- back-propagate through time (BPTT)
   for step=self.rho,1,-1 do
      self.module:accUpdateGradParameters(input, gradOutput[step], lr)
   end
   ]=]
   self.module:accUpdateGradParameters(input, gradOutput, lr)

end

--[=[
function Repeater:maxBPTTstep(rho)
   self.rho = rho
   self.module:maxBPTTstep(rho)
end


function Repeater:__tostring__()
   local tab = '  '
   local line = '\n'
   local str = torch.type(self) .. ' {' .. line
   str = str .. tab .. '[  input,    input,  ...,  input  ]'.. line
   str = str .. tab .. '     V         V             V     '.. line
   str = str .. tab .. tostring(self.modules[1]):gsub(line, line .. tab) .. line
   str = str .. tab .. '     V         V             V     '.. line
   str = str .. tab .. '[output(1),output(2),...,output('..self.rho..')]' .. line
   str = str .. '}'
   return str
end
]=]
